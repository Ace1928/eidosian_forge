from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
class ScalarAxis(Axis):
    """
    Along this axis of the CIFTI-2 vector/matrix each row/column has been given
    a unique name and optionally metadata
    """

    def __init__(self, name, meta=None):
        """
        Parameters
        ----------
        name : array_like
            (N, ) string array with the parcel names
        meta :  array_like
            (N, ) object array with a dictionary of metadata for each row/column.
            Defaults to empty dictionary
        """
        self.name = np.asanyarray(name, dtype='U')
        if meta is None:
            meta = [{} for _ in range(self.name.size)]
        self.meta = np.asanyarray(meta, dtype='object')
        for check_name in ('name', 'meta'):
            if getattr(self, check_name).shape != (self.size,):
                raise ValueError(f'Input {check_name} has incorrect shape ({getattr(self, check_name).shape}) for ScalarAxis axis')

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new Scalar axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        ScalarAxis
        """
        names = [nm.map_name for nm in mim.named_maps]
        meta = [{} if nm.metadata is None else dict(nm.metadata) for nm in mim.named_maps]
        return cls(names, meta)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`.cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SCALARS')
        for name, meta in zip(self.name, self.meta):
            named_map = cifti2.Cifti2NamedMap(name, cifti2.Cifti2MetaData(meta))
            mim.append(named_map)
        return mim

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        """
        Compares two Scalars

        Parameters
        ----------
        other : ScalarAxis
            scalar axis to be compared

        Returns
        -------
        bool : False if type, length or content do not match
        """
        if not isinstance(other, ScalarAxis) or self.size != other.size:
            return False
        return np.array_equal(self.name, other.name) and np.array_equal(self.meta, other.meta)

    def __add__(self, other):
        """
        Concatenates two Scalars

        Parameters
        ----------
        other : ScalarAxis
            scalar axis to be appended to the current one

        Returns
        -------
        ScalarAxis
        """
        if not isinstance(other, ScalarAxis):
            return NotImplemented
        return ScalarAxis(np.append(self.name, other.name), np.append(self.meta, other.meta))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_element(item)
        return self.__class__(self.name[item], self.meta[item])

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the row/column
        - dictionary with the element metadata
        """
        return (self.name[index], self.meta[index])