import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
class GridDOSCollection(DOSCollection):

    def __init__(self, dos_series: Iterable[GridDOSData], energies: Optional[Sequence[float]]=None) -> None:
        dos_list = list(dos_series)
        if energies is None:
            if len(dos_list) == 0:
                raise ValueError('Must provide energies to create a GridDOSCollection without any DOS data.')
            self._energies = dos_list[0].get_energies()
        else:
            self._energies = np.asarray(energies)
        self._weights = np.empty((len(dos_list), len(self._energies)), float)
        self._info = []
        for i, dos_data in enumerate(dos_list):
            if not isinstance(dos_data, GridDOSData):
                raise TypeError('GridDOSCollection can only store GridDOSData objects.')
            if dos_data.get_energies().shape != self._energies.shape or not np.allclose(dos_data.get_energies(), self._energies):
                raise ValueError('All GridDOSData objects in GridDOSCollection must have the same energy axis.')
            self._weights[i, :] = dos_data.get_weights()
            self._info.append(dos_data.info)

    def get_energies(self) -> Sequence[float]:
        return self._energies.copy()

    def get_all_weights(self) -> Sequence[Sequence[float]]:
        return self._weights.copy()

    def __len__(self) -> int:
        return self._weights.shape[0]

    @overload
    def __getitem__(self, item: int) -> DOSData:
        ...

    @overload
    def __getitem__(self, item: slice) -> 'GridDOSCollection':
        ...

    def __getitem__(self, item):
        if isinstance(item, int):
            return GridDOSData(self._energies, self._weights[item, :], info=self._info[item])
        elif isinstance(item, slice):
            return type(self)([self[i] for i in range(len(self))[item]])
        else:
            raise TypeError('index in DOSCollection must be an integer or slice')

    @classmethod
    def from_data(cls, energies: Sequence[float], weights: Sequence[Sequence[float]], info: Sequence[Info]=None) -> 'GridDOSCollection':
        """Create a GridDOSCollection from data with a common set of energies

        This convenience method may also be more efficient as it limits
        redundant copying/checking of the data.

        Args:
            energies: common set of energy values for input data
            weights: array of DOS weights with rows corresponding to different
                datasets
            info: sequence of info dicts corresponding to weights rows.

        Returns:
            Collection of DOS data (in RawDOSData format)
        """
        weights_array = np.asarray(weights, dtype=float)
        if len(weights_array.shape) != 2:
            raise IndexError('Weights must be a 2-D array or nested sequence')
        if weights_array.shape[0] < 1:
            raise IndexError('Weights cannot be empty')
        if weights_array.shape[1] != len(energies):
            raise IndexError('Length of weights rows must equal size of x')
        info = cls._check_weights_and_info(weights, info)
        dos_collection = cls([GridDOSData(energies, weights_array[0])])
        dos_collection._weights = weights_array
        dos_collection._info = list(info)
        return dos_collection

    def select(self, **info_selection: str) -> 'DOSCollection':
        """Narrow GridDOSCollection to items with specified info

        For example, if ::

          dc = GridDOSCollection([GridDOSData(x, y1,
                                              info={'a': '1', 'b': '1'}),
                                  GridDOSData(x, y2,
                                              info={'a': '2', 'b': '1'})])

        then ::

          dc.select(b='1')

        will return an identical object to dc, while ::

          dc.select(a='1')

        will return a DOSCollection with only the first item and ::

          dc.select(a='2', b='1')

        will return a DOSCollection with only the second item.

        """
        matches = self._select_to_list(self, info_selection)
        if len(matches) == 0:
            return type(self)([], energies=self._energies)
        else:
            return type(self)(matches)

    def select_not(self, **info_selection: str) -> 'DOSCollection':
        """Narrow GridDOSCollection to items without specified info

        For example, if ::

          dc = GridDOSCollection([GridDOSData(x, y1,
                                              info={'a': '1', 'b': '1'}),
                                  GridDOSData(x, y2,
                                              info={'a': '2', 'b': '1'})])

        then ::

          dc.select_not(b='2')

        will return an identical object to dc, while ::

          dc.select_not(a='2')

        will return a DOSCollection with only the first item and ::

          dc.select_not(a='1', b='1')

        will return a DOSCollection with only the second item.

        """
        matches = self._select_to_list(self, info_selection, negative=True)
        if len(matches) == 0:
            return type(self)([], energies=self._energies)
        else:
            return type(self)(matches)

    def plot(self, npts: int=0, xmin: float=None, xmax: float=None, width: float=None, smearing: str='Gauss', ax: 'matplotlib.axes.Axes'=None, show: bool=False, filename: str=None, mplargs: dict=None) -> 'matplotlib.axes.Axes':
        """Simple plot of collected DOS data, resampled onto a grid

        If the special key 'label' is present in self.info, this will be set
        as the label for the plotted line (unless overruled in mplargs). The
        label is only seen if a legend is added to the plot (i.e. by calling
        `ax.legend()`).

        Args:
            npts:
                Number of points in resampled x-axis. If set to zero (default),
                no resampling is performed and the stored data is plotted
                directly.
            xmin, xmax:
                output data range; this limits the resampling range as well as
                the plotting output
            width: Width of broadening kernel, passed to self.sample()
            smearing: selection of broadening kernel, passed to self.sample()
            ax: existing Matplotlib axes object. If not provided, a new figure
                with one set of axes will be created using Pyplot
            show: show the figure on-screen
            filename: if a path is given, save the figure to this file
            mplargs: additional arguments to pass to matplotlib plot command
                (e.g. {'linewidth': 2} for a thicker line).

        Returns:
            Plotting axes. If "ax" was set, this is the same object.
        """
        npts, width = GridDOSData._interpret_smearing_args(npts, width)
        if npts:
            assert isinstance(width, float)
            dos = self.sample_grid(npts, xmin=xmin, xmax=xmax, width=width, smearing=smearing)
        else:
            dos = self
        energies, all_y = (dos._energies, dos._weights)
        all_labels = [DOSData.label_from_info(data.info) for data in self]
        with SimplePlottingAxes(ax=ax, show=show, filename=filename) as ax:
            self._plot_broadened(ax, energies, all_y, all_labels, mplargs)
        return ax

    @staticmethod
    def _plot_broadened(ax: 'matplotlib.axes.Axes', energies: Sequence[float], all_y: np.ndarray, all_labels: Sequence[str], mplargs: Union[Dict, None]):
        """Plot DOS data with labels to axes

        This is separated into another function so that subclasses can
        manipulate broadening, labels etc in their plot() method."""
        if mplargs is None:
            mplargs = {}
        all_lines = ax.plot(energies, all_y.T, **mplargs)
        for line, label in zip(all_lines, all_labels):
            line.set_label(label)
        ax.legend()
        ax.set_xlim(left=min(energies), right=max(energies))
        ax.set_ylim(bottom=0)