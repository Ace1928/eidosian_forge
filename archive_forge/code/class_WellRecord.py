import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
class WellRecord:
    """WellRecord stores all time course signals of a phenotype Microarray well.

    The single time points and signals can be accessed iterating on the
    WellRecord or using lists indexes or slices:

    >>> from Bio import phenotype
    >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")
    >>> well = plate['A05']
    >>> for time, signal in well:
    ...    print("Time: %f, Signal: %f" % (time, signal)) # doctest:+ELLIPSIS
    ...
    Time: 0.000000, Signal: 14.000000
    Time: 0.250000, Signal: 13.000000
    Time: 0.500000, Signal: 15.000000
    Time: 0.750000, Signal: 15.000000
    ...
    >>> well[1]
    16.0
    >>> well[1:5]
    [16.0, 20.0, 18.0, 15.0]
    >>> well[1:5:0.5]
    [16.0, 19.0, 20.0, 18.0, 18.0, 18.0, 15.0, 18.0]

    If a time point was not present in the input file but it's between the
    minimum and maximum time point, the interpolated signal is returned,
    otherwise a nan value:

    >>> well[1.3]
    19.0
    >>> well[1250]
    nan

    Two WellRecord objects can be compared: if their input time/signal pairs
    are exactly the same, the two records are considered equal:

    >>> well2 = plate['H12']
    >>> well == well2
    False

    Two WellRecord objects can be summed up or subtracted from each other: a new
    WellRecord object is returned, having the left operand id.

    >>> well1 = plate['A05']
    >>> well2 = well + well1
    >>> print(well2.id)
    A05

    If SciPy is installed, a sigmoid function can be fitted to the PM curve,
    in order to extract some parameters; three sigmoid functions are available:
    * gompertz
    * logistic
    * richards
    The functions are described in Zwietering et al., 1990 (PMID: 16348228)

    For example::

        well.fit()
        print(well.slope, well.model)
        (61.853516785566917, 'logistic')

    If not sigmoid function is specified, the first one that is successfully
    fitted is used. The user can also specify a specific function.

    To specify gompertz::

        well.fit('gompertz')
        print(well.slope, well.model)
        (127.94630059171354, 'gompertz')

    If no function can be fitted, the parameters are left as None, except for
    the max, min, average_height and area.
    """

    def __init__(self, wellid, plate=None, signals=None):
        """Initialize the class."""
        if plate is None:
            self.plate = PlateRecord(None)
        else:
            self.plate = plate
        self.id = wellid
        self.max = None
        self.min = None
        self.average_height = None
        self.area = None
        self.plateau = None
        self.slope = None
        self.lag = None
        self.v = None
        self.y0 = None
        self.model = None
        if signals is None:
            self._signals = {}
        else:
            self._signals = signals

    def _interpolate(self, time):
        """Linear interpolation of the signals at certain time points (PRIVATE)."""
        times = sorted(self._signals.keys())
        return np.interp(time, times, [self._signals[x] for x in times], left=np.nan, right=np.nan)

    def __setitem__(self, time, signal):
        """Assign a signal at a certain time point."""
        try:
            time = float(time)
        except ValueError:
            raise ValueError('Time point should be a number')
        try:
            signal = float(signal)
        except ValueError:
            raise ValueError('Signal should be a number')
        self._signals[time] = signal

    def __getitem__(self, time):
        """Return a subset of signals or a single signal."""
        if isinstance(time, slice):
            if time.start is None:
                start = 0
            else:
                start = time.start
            if time.stop is None:
                stop = max(self.get_times())
            else:
                stop = time.stop
            time = np.arange(start, stop, time.step)
            return list(self._interpolate(time))
        elif isinstance(time, (float, int)):
            return self._interpolate(time)
        raise ValueError('Invalid index')

    def __iter__(self):
        for time in sorted(self._signals.keys()):
            yield (time, self._signals[time])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if list(self._signals.keys()) != list(other._signals.keys()):
                return False
            for k in self._signals:
                if np.isnan(self[k]) and np.isnan(other[k]):
                    continue
                elif self[k] != other[k]:
                    return False
            return True
        else:
            return False

    def __add__(self, well):
        """Add another WellRecord object.

        A new WellRecord object is returned, having the same id as the
        left operand
        """
        if not isinstance(well, WellRecord):
            raise TypeError('Expecting a WellRecord object')
        signals = {}
        times = set(self._signals.keys()).union(set(well._signals.keys()))
        for t in sorted(times):
            signals[t] = self[t] + well[t]
        neww = WellRecord(self.id, signals=signals)
        return neww

    def __sub__(self, well):
        """Subtract another WellRecord object.

        A new WellRecord object is returned, having the same id as the
        left operand
        """
        if not isinstance(well, WellRecord):
            raise TypeError('Expecting a WellRecord object')
        signals = {}
        times = set(self._signals.keys()).union(set(well._signals.keys()))
        for t in sorted(times):
            signals[t] = self[t] - well[t]
        neww = WellRecord(self.id, signals=signals)
        return neww

    def __len__(self):
        """Return the number of time points sampled."""
        return len(self._signals)

    def __repr__(self):
        """Return a (truncated) representation of the signals for debugging."""
        if len(self) > 7:
            return "%s('%s, ..., %s')" % (self.__class__.__name__, ', '.join([str(x) for x in self.get_raw()[:5]]), str(self.get_raw()[-1]))
        else:
            return '%s(%s)' % (self.__class__.__name__, ', '.join([str(x) for x in self.get_raw()]))

    def __str__(self):
        """Return a human readable summary of the record (string).

        The python built-in function str works by calling the object's __str__
        method.  e.g.

        >>> from Bio import phenotype
        >>> plate = phenotype.read("phenotype/Plate.json", "pm-json")
        >>> record = plate['A05']
        >>> print(record)
        Plate ID: PM01
        Well ID: A05
        Time points: 384
        Minum signal 0.25 at time 13.00
        Maximum signal 19.50 at time 23.00
        WellRecord('(0.0, 14.0), (0.25, 13.0), (0.5, 15.0), (0.75, 15.0), (1.0, 16.0), ..., (95.75, 16.0)')

        Note that long time spans are shown truncated.
        """
        lines = []
        if self.plate and self.plate.id:
            lines.append(f'Plate ID: {self.plate.id}')
        if self.id:
            lines.append(f'Well ID: {self.id}')
        lines.append('Time points: %i' % len(self))
        lines.append('Minum signal %.2f at time %.2f' % min(self, key=lambda x: x[1]))
        lines.append('Maximum signal %.2f at time %.2f' % max(self, key=lambda x: x[1]))
        lines.append(repr(self))
        return '\n'.join(lines)

    def get_raw(self):
        """Get a list of time/signal pairs."""
        return [(t, self._signals[t]) for t in sorted(self._signals.keys())]

    def get_times(self):
        """Get a list of the recorded time points."""
        return sorted(self._signals.keys())

    def get_signals(self):
        """Get a list of the recorded signals (ordered by collection time)."""
        return [self._signals[t] for t in sorted(self._signals.keys())]

    def fit(self, function=('gompertz', 'logistic', 'richards')):
        """Fit a sigmoid function to this well and extract curve parameters.

        If function is None or an empty tuple/list, then no fitting is done.
        Only the object's ``.min``, ``.max`` and ``.average_height`` are
        calculated.

        By default the following fitting functions will be used in order:
         - gompertz
         - logistic
         - richards

        The first function that is successfully fitted to the signals will
        be used to extract the curve parameters and update ``.area`` and
        ``.model``. If no function can be fitted an exception is raised.

        The function argument should be a tuple or list of any of these three
        function names as strings.

        There is no return value.
        """
        avail_func = ('gompertz', 'logistic', 'richards')
        self.max = max(self, key=lambda x: x[1])[1]
        self.min = min(self, key=lambda x: x[1])[1]
        self.average_height = np.array(self.get_signals()).mean()
        if not function:
            self.area = None
            self.model = None
            return
        for sigmoid_func in function:
            if sigmoid_func not in avail_func:
                raise ValueError(f'Fitting function {sigmoid_func!r} not supported')
        from .pm_fitting import fit, get_area
        from .pm_fitting import logistic, gompertz, richards
        function_map = {'logistic': logistic, 'gompertz': gompertz, 'richards': richards}
        self.area = get_area(self.get_signals(), self.get_times())
        self.model = None
        for sigmoid_func in function:
            func = function_map[sigmoid_func]
            try:
                (self.plateau, self.slope, self.lag, self.v, self.y0), pcov = fit(func, self.get_times(), self.get_signals())
                self.model = sigmoid_func
                return
            except RuntimeError:
                continue
        raise RuntimeError('Could not fit any sigmoid function')