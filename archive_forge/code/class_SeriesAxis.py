from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
class SeriesAxis(Axis):
    """
    Along this axis of the CIFTI-2 vector/matrix the rows/columns increase monotonously in time

    This Axis describes the time point of each row/column.
    """
    size = None

    def __init__(self, start, step, size, unit='SECOND'):
        """
        Creates a new SeriesAxis axis

        Parameters
        ----------
        start : float
            starting time point
        step :  float
            sampling time (TR)
        size : int
            number of time points
        unit : str
            Unit of the step size (one of 'second', 'hertz', 'meter', or 'radian')
        """
        self.unit = unit
        self.start = start
        self.step = step
        self.size = size

    @property
    def time(self):
        return np.arange(self.size) * self.step + self.start

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new SeriesAxis axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        SeriesAxis
        """
        start = mim.series_start * 10 ** mim.series_exponent
        step = mim.series_step * 10 ** mim.series_exponent
        return cls(start, step, mim.number_of_series_points, mim.series_unit)

    def to_mapping(self, dim):
        """
        Converts the SeriesAxis to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SERIES')
        mim.series_exponent = 0
        mim.series_start = self.start
        mim.series_step = self.step
        mim.number_of_series_points = self.size
        mim.series_unit = self.unit
        return mim
    _unit = None

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value.upper() not in ('SECOND', 'HERTZ', 'METER', 'RADIAN'):
            raise ValueError('SeriesAxis unit should be one of ' + "('second', 'hertz', 'meter', or 'radian'")
        self._unit = value.upper()

    def __len__(self):
        return self.size

    def __eq__(self, other):
        """
        True if start, step, size, and unit are the same.
        """
        return isinstance(other, SeriesAxis) and self.start == other.start and (self.step == other.step) and (self.size == other.size) and (self.unit == other.unit)

    def __add__(self, other):
        """
        Concatenates two SeriesAxis

        Parameters
        ----------
        other : SeriesAxis
            Time SeriesAxis to append at the end of the current time SeriesAxis.
            Note that the starting time of the other time SeriesAxis is ignored.

        Returns
        -------
        SeriesAxis
            New time SeriesAxis with the concatenation of the two

        Raises
        ------
        ValueError
            raised if the repetition time of the two time SeriesAxis is different
        """
        if isinstance(other, SeriesAxis):
            if other.step != self.step:
                raise ValueError('Can only concatenate SeriesAxis with the same step size')
            if other.unit != self.unit:
                raise ValueError('Can only concatenate SeriesAxis with the same unit')
            return SeriesAxis(self.start, self.step, self.size + other.size, self.unit)
        return NotImplemented

    def __getitem__(self, item):
        if isinstance(item, slice):
            step = 1 if item.step is None else item.step
            idx_start = (self.size - 1 if step < 0 else 0) if item.start is None else item.start if item.start >= 0 else self.size + item.start
            idx_end = (-1 if step < 0 else self.size) if item.stop is None else item.stop if item.stop >= 0 else self.size + item.stop
            if idx_start > self.size and step < 0:
                idx_start = self.size - 1
            if idx_end > self.size:
                idx_end = self.size
            nelements = (idx_end - idx_start) // step
            if nelements < 0:
                nelements = 0
            return SeriesAxis(idx_start * self.step + self.start, self.step * step, nelements, self.unit)
        elif isinstance(item, int):
            return self.get_element(item)
        raise IndexError('SeriesAxis can only be indexed with integers or slices without breaking the regular structure')

    def get_element(self, index):
        """
        Gives the time point of a specific row/column

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        float
        """
        original_index = index
        if index < 0:
            index = self.size + index
        if index >= self.size or index < 0:
            raise IndexError('index %i is out of range for SeriesAxis with size %i' % (original_index, self.size))
        return self.start + self.step * index