import warnings
from .. import h5ds
from ..h5py_warnings import H5pyDeprecationWarning
from . import base
from .base import phil, with_phil
from .dataset import Dataset
class DimensionManager(base.CommonStateObject):
    """
        Represents a collection of dimension associated with a dataset.

        Like AttributeManager, an instance of this class is returned when
        accessing the ".dims" property on a Dataset.
    """

    @with_phil
    def __init__(self, parent):
        """ Private constructor.
        """
        self._id = parent.id

    @with_phil
    def __getitem__(self, index):
        """ Return a Dimension object
        """
        if index > len(self) - 1:
            raise IndexError('Index out of range')
        return DimensionProxy(self._id, index)

    @with_phil
    def __len__(self):
        """ Number of dimensions associated with the dataset. """
        return self._id.rank

    @with_phil
    def __iter__(self):
        """ Iterate over the dimensions. """
        for i in range(len(self)):
            yield self[i]

    @with_phil
    def __repr__(self):
        if not self._id:
            return '<Dimensions of closed HDF5 dataset>'
        return '<Dimensions of HDF5 object at %s>' % id(self._id)

    def create_scale(self, dset, name=''):
        """ Create a new dimension, from an initial scale.

        Provide the dataset and a name for the scale.
        """
        warnings.warn('other_ds.dims.create_scale(ds, name) is deprecated. Use ds.make_scale(name) instead.', H5pyDeprecationWarning, stacklevel=2)
        dset.make_scale(name)