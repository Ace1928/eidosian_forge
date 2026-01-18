import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
class _PlateCarreeFormatter(Formatter):
    """
    Base class for formatting ticks on geographical axes using a
    rectangular projection (e.g. Plate Carree, Mercator).

    """

    def __init__(self, direction_label=True, degree_symbol='°', number_format='g', transform_precision=1e-08, dms=False, minute_symbol='′', second_symbol='″', seconds_number_format='g', auto_hide=True, decimal_point=None, cardinal_labels=None):
        """
        Base class for simpler implementation of specialised formatters
        for latitude and longitude axes.

        """
        self._direction_labels = direction_label
        self._degree_symbol = degree_symbol
        self._degrees_number_format = number_format
        self._transform_precision = transform_precision
        self._dms = dms
        self._minute_symbol = minute_symbol
        self._second_symbol = second_symbol
        self._seconds_num_format = seconds_number_format
        self._auto_hide = auto_hide
        self._auto_hide_degrees = False
        self._auto_hide_minutes = False
        self._precision = 5
        if decimal_point is None and mpl.rcParams['axes.formatter.use_locale']:
            import locale
            decimal_point = locale.localeconv()['decimal_point']
        if cardinal_labels is None:
            cardinal_labels = {}
        self._cardinal_labels = cardinal_labels
        self._decimal_point = decimal_point
        self._source_projection = None
        self._target_projection = None

    def __call__(self, value, pos=None):
        if self._source_projection is not None:
            projected_value = self._apply_transform(value, self._target_projection, self._source_projection)
            f = 1.0 / self._transform_precision
            projected_value = round(f * projected_value) / f
        else:
            projected_value = value
        return self._format_value(projected_value, value)

    def _format_value(self, value, original_value):
        hemisphere = ''
        sign = ''
        if self._direction_labels:
            hemisphere = self._hemisphere(value, original_value)
        elif value != 0 and self._hemisphere(value, original_value) in ['W', 'S']:
            sign = '-'
        if not self._dms:
            return sign + self._format_degrees(abs(value)) + hemisphere
        value, deg, mn, sec = self._get_dms(abs(value))
        label = ''
        if sec:
            label = self._format_seconds(sec)
        if mn or (not self._auto_hide_minutes and label):
            label = self._format_minutes(mn) + label
        if not self._auto_hide_degrees or not label:
            label = sign + self._format_degrees(deg) + label + hemisphere
        return label

    def _get_dms(self, x):
        """Convert to degrees, minutes, seconds

        Parameters
        ----------
        x: float or array of floats
            Degrees

        Return
        ------
        x: degrees rounded to the requested precision
        degs: degrees
        mins: minutes
        secs: seconds
        """
        self._precision = 6
        x = np.asarray(x, 'd')
        degs = np.round(x, self._precision).astype('i')
        y = (x - degs) * 60
        mins = np.round(y, self._precision).astype('i')
        secs = np.round((y - mins) * 60, self._precision - 3)
        return (x, degs, mins, secs)

    def set_axis(self, axis):
        super().set_axis(axis)
        if self.axis is None or not isinstance(self.axis.axes, GeoAxes):
            self._source_projection = None
            self._target_projection = None
            return
        self._source_projection = self.axis.axes.projection
        if not isinstance(self._source_projection, (ccrs._RectangularProjection, ccrs.Mercator)):
            raise TypeError('This formatter cannot be used with non-rectangular projections.')
        self._target_projection = ccrs.PlateCarree(globe=self._source_projection.globe)

    def set_locs(self, locs):
        Formatter.set_locs(self, locs)
        if not self._auto_hide:
            return
        self.locs, degs, mins, secs = self._get_dms(self.locs)
        secs = np.round(secs, self._precision - 3).astype('i')
        secs0 = secs == 0
        mins0 = mins == 0

        def auto_hide(valid, values):
            """Should I switch on auto_hide?"""
            if not valid.any():
                return False
            if valid.sum() == 1:
                return True
            return np.diff(values.compress(valid)).max() == 1
        self._auto_hide_minutes = auto_hide(secs0, mins)
        self._auto_hide_degrees = auto_hide(secs0 & mins0, degs)

    def _format_degrees(self, deg):
        """Format degrees as an integer"""
        if self._dms:
            deg = int(deg)
            number_format = 'd'
        else:
            number_format = self._degrees_number_format
        value = f'{abs(deg):{number_format}}{self._degree_symbol}'
        if self._decimal_point is not None:
            value = value.replace('.', self._decimal_point)
        return value

    def _format_minutes(self, mn):
        """Format minutes as an integer"""
        return f'{int(mn):d}{self._minute_symbol}'

    def _format_seconds(self, sec):
        """Format seconds as an float"""
        return f'{sec:{self._seconds_num_format}}{self._second_symbol}'

    def _apply_transform(self, value, target_proj, source_crs):
        """
        Given a single value, a target projection and a source CRS,
        transform the value from the source CRS to the target
        projection, returning a single value.

        """
        raise NotImplementedError('A subclass must implement this method.')

    def _hemisphere(self, value, value_source_crs):
        """
        Given both a tick value in the Plate Carree projection and the
        same value in the source CRS, return a string indicating the
        hemisphere that the value is in.

        Must be over-ridden by the derived class.

        """
        raise NotImplementedError('A subclass must implement this method.')