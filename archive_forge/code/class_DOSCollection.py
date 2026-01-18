import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
class DOSCollection(collections.abc.Sequence):
    """Base class for a collection of DOSData objects"""

    def __init__(self, dos_series: Iterable[DOSData]) -> None:
        self._data = list(dos_series)

    def _sample(self, energies: Sequence[float], width: float=0.1, smearing: str='Gauss') -> np.ndarray:
        """Sample the DOS data at chosen points, with broadening

        This samples the underlying DOS data in the same way as the .sample()
        method of those DOSData items, returning a 2-D array with columns
        corresponding to x and rows corresponding to the collected data series.

        Args:
            energies: energy values for sampling
            width: Width of broadening kernel
            smearing: selection of broadening kernel (only "Gauss" is currently
                supported)

        Returns:
            Weights sampled from a broadened DOS at values corresponding to x,
            in rows corresponding to DOSData entries contained in this object
        """
        if len(self) == 0:
            raise IndexError('No data to sample')
        return np.asarray([data._sample(energies, width=width, smearing=smearing) for data in self])

    def plot(self, npts: int=1000, xmin: float=None, xmax: float=None, width: float=0.1, smearing: str='Gauss', ax: 'matplotlib.axes.Axes'=None, show: bool=False, filename: str=None, mplargs: dict=None) -> 'matplotlib.axes.Axes':
        """Simple plot of collected DOS data, resampled onto a grid

        If the special key 'label' is present in self.info, this will be set
        as the label for the plotted line (unless overruled in mplargs). The
        label is only seen if a legend is added to the plot (i.e. by calling
        `ax.legend()`).

        Args:
            npts, xmin, xmax: output data range, as passed to self.sample_grid
            width: Width of broadening kernel, passed to self.sample_grid()
            smearing: selection of broadening kernel for self.sample_grid()
            ax: existing Matplotlib axes object. If not provided, a new figure
                with one set of axes will be created using Pyplot
            show: show the figure on-screen
            filename: if a path is given, save the figure to this file
            mplargs: additional arguments to pass to matplotlib plot command
                (e.g. {'linewidth': 2} for a thicker line).

        Returns:
            Plotting axes. If "ax" was set, this is the same object.
        """
        return self.sample_grid(npts, xmin=xmin, xmax=xmax, width=width, smearing=smearing).plot(npts=npts, xmin=xmin, xmax=xmax, width=width, smearing=smearing, ax=ax, show=show, filename=filename, mplargs=mplargs)

    def sample_grid(self, npts: int, xmin: float=None, xmax: float=None, padding: float=3, width: float=0.1, smearing: str='Gauss') -> 'GridDOSCollection':
        """Sample the DOS data on an evenly-spaced energy grid

        Args:
            npts: Number of sampled points
            xmin: Minimum sampled energy value; if unspecified, a default is
                chosen
            xmax: Maximum sampled energy value; if unspecified, a default is
                chosen
            padding: If xmin/xmax is unspecified, default value will be padded
                by padding * width to avoid cutting off peaks.
            width: Width of broadening kernel, passed to self.sample_grid()
            smearing: selection of broadening kernel, for self.sample_grid()

        Returns:
            (energy values, sampled DOS)
        """
        if len(self) == 0:
            raise IndexError('No data to sample')
        if xmin is None:
            xmin = min((min(data.get_energies()) for data in self)) - padding * width
        if xmax is None:
            xmax = max((max(data.get_energies()) for data in self)) + padding * width
        return GridDOSCollection([data.sample_grid(npts, xmin=xmin, xmax=xmax, width=width, smearing=smearing) for data in self])

    @classmethod
    def from_data(cls, energies: Sequence[float], weights: Sequence[Sequence[float]], info: Sequence[Info]=None) -> 'DOSCollection':
        """Create a DOSCollection from data sharing a common set of energies

        This is a convenience method to be used when all the DOS data in the
        collection has a common energy axis. There is no performance advantage
        in using this method for the generic DOSCollection, but for
        GridDOSCollection it is more efficient.

        Args:
            energy: common set of energy values for input data
            weights: array of DOS weights with rows corresponding to different
                datasets
            info: sequence of info dicts corresponding to weights rows.

        Returns:
            Collection of DOS data (in RawDOSData format)
        """
        info = cls._check_weights_and_info(weights, info)
        return cls((RawDOSData(energies, row_weights, row_info) for row_weights, row_info in zip(weights, info)))

    @staticmethod
    def _check_weights_and_info(weights: Sequence[Sequence[float]], info: Union[Sequence[Info], None]) -> Sequence[Info]:
        if info is None:
            info = [{} for _ in range(len(weights))]
        elif len(info) != len(weights):
            raise ValueError('Length of info must match number of rows in weights')
        return info

    @overload
    def __getitem__(self, item: int) -> DOSData:
        ...

    @overload
    def __getitem__(self, item: slice) -> 'DOSCollection':
        ...

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._data[item]
        elif isinstance(item, slice):
            return type(self)(self._data[item])
        else:
            raise TypeError('index in DOSCollection must be an integer or slice')

    def __len__(self) -> int:
        return len(self._data)

    def _almost_equals(self, other: Any) -> bool:
        """Compare with another DOSCollection for testing purposes"""
        if not isinstance(other, type(self)):
            return False
        elif not len(self) == len(other):
            return False
        else:
            return all([a._almost_equals(b) for a, b in zip(self, other)])

    def total(self) -> DOSData:
        """Sum all the DOSData in this Collection and label it as 'Total'"""
        data = self.sum_all()
        data.info.update({'label': 'Total'})
        return data

    def sum_all(self) -> DOSData:
        """Sum all the DOSData contained in this Collection"""
        if len(self) == 0:
            raise IndexError('No data to sum')
        elif len(self) == 1:
            data = self[0].copy()
        else:
            data = reduce(lambda x, y: x + y, self)
        return data
    D = TypeVar('D', bound=DOSData)

    @staticmethod
    def _select_to_list(dos_collection: Sequence[D], info_selection: Dict[str, str], negative: bool=False) -> List[D]:
        query = set(info_selection.items())
        if negative:
            return [data for data in dos_collection if not query.issubset(set(data.info.items()))]
        else:
            return [data for data in dos_collection if query.issubset(set(data.info.items()))]

    def select(self, **info_selection: str) -> 'DOSCollection':
        """Narrow DOSCollection to items with specified info

        For example, if ::

          dc = DOSCollection([DOSData(x1, y1, info={'a': '1', 'b': '1'}),
                              DOSData(x2, y2, info={'a': '2', 'b': '1'})])

        then ::

          dc.select(b='1')

        will return an identical object to dc, while ::

          dc.select(a='1')

        will return a DOSCollection with only the first item and ::

          dc.select(a='2', b='1')

        will return a DOSCollection with only the second item.

        """
        matches = self._select_to_list(self, info_selection)
        return type(self)(matches)

    def select_not(self, **info_selection: str) -> 'DOSCollection':
        """Narrow DOSCollection to items without specified info

        For example, if ::

          dc = DOSCollection([DOSData(x1, y1, info={'a': '1', 'b': '1'}),
                              DOSData(x2, y2, info={'a': '2', 'b': '1'})])

        then ::

          dc.select_not(b='2')

        will return an identical object to dc, while ::

          dc.select_not(a='2')

        will return a DOSCollection with only the first item and ::

          dc.select_not(a='1', b='1')

        will return a DOSCollection with only the second item.

        """
        matches = self._select_to_list(self, info_selection, negative=True)
        return type(self)(matches)

    def sum_by(self, *info_keys: str) -> 'DOSCollection':
        """Return a DOSCollection with some data summed by common attributes

        For example, if ::

          dc = DOSCollection([DOSData(x1, y1, info={'a': '1', 'b': '1'}),
                              DOSData(x2, y2, info={'a': '2', 'b': '1'}),
                              DOSData(x3, y3, info={'a': '2', 'b': '2'})])

        then ::

          dc.sum_by('b')

        will return a collection equivalent to ::

          DOSCollection([DOSData(x1, y1, info={'a': '1', 'b': '1'})
                         + DOSData(x2, y2, info={'a': '2', 'b': '1'}),
                         DOSData(x3, y3, info={'a': '2', 'b': '2'})])

        where the resulting contained DOSData have info attributes of
        {'b': '1'} and {'b': '2'} respectively.

        dc.sum_by('a', 'b') on the other hand would return the full three-entry
        collection, as none of the entries have common 'a' *and* 'b' info.

        """

        def _matching_info_tuples(data: DOSData):
            """Get relevent dict entries in tuple form

            e.g. if data.info = {'a': 1, 'b': 2, 'c': 3}
                 and info_keys = ('a', 'c')

                 then return (('a', 1), ('c': 3))
            """
            matched_keys = set(info_keys) & set(data.info)
            return tuple(sorted([(key, data.info[key]) for key in matched_keys]))
        all_combos = map(_matching_info_tuples, self)
        unique_combos = sorted(set(all_combos))
        collection_data = [self.select(**dict(combo)).sum_all() for combo in unique_combos]
        return type(self)(collection_data)

    def __add__(self, other: Union['DOSCollection', DOSData]) -> 'DOSCollection':
        """Join entries between two DOSCollection objects of the same type

        It is also possible to add a single DOSData object without wrapping it
        in a new collection: i.e. ::

          DOSCollection([dosdata1]) + DOSCollection([dosdata2])

        or ::

          DOSCollection([dosdata1]) + dosdata2

        will return ::

          DOSCollection([dosdata1, dosdata2])

        """
        return _add_to_collection(other, self)