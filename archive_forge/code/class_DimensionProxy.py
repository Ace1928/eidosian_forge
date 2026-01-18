import warnings
from .. import h5ds
from ..h5py_warnings import H5pyDeprecationWarning
from . import base
from .base import phil, with_phil
from .dataset import Dataset
class DimensionProxy(base.CommonStateObject):
    """
        Represents an HDF5 "dimension".
    """

    @property
    @with_phil
    def label(self):
        """ Get or set the dimension scale label """
        return self._d(h5ds.get_label(self._id, self._dimension))

    @label.setter
    @with_phil
    def label(self, val):
        h5ds.set_label(self._id, self._dimension, self._e(val))

    @with_phil
    def __init__(self, id_, dimension):
        self._id = id_
        self._dimension = dimension

    @with_phil
    def __hash__(self):
        return hash((type(self), self._id, self._dimension))

    @with_phil
    def __eq__(self, other):
        return hash(self) == hash(other)

    @with_phil
    def __iter__(self):
        for k in self.keys():
            yield k

    @with_phil
    def __len__(self):
        return h5ds.get_num_scales(self._id, self._dimension)

    @with_phil
    def __getitem__(self, item):
        if isinstance(item, int):
            scales = []
            h5ds.iterate(self._id, self._dimension, scales.append, 0)
            return Dataset(scales[item])
        else:

            def f(dsid):
                """ Iterate over scales to find a matching name """
                if h5ds.get_scale_name(dsid) == self._e(item):
                    return dsid
            res = h5ds.iterate(self._id, self._dimension, f, 0)
            if res is None:
                raise KeyError(item)
            return Dataset(res)

    def attach_scale(self, dset):
        """ Attach a scale to this dimension.

        Provide the Dataset of the scale you would like to attach.
        """
        with phil:
            h5ds.attach_scale(self._id, dset.id, self._dimension)

    def detach_scale(self, dset):
        """ Remove a scale from this dimension.

        Provide the Dataset of the scale you would like to remove.
        """
        with phil:
            h5ds.detach_scale(self._id, dset.id, self._dimension)

    def items(self):
        """ Get a list of (name, Dataset) pairs with all scales on this
        dimension.
        """
        with phil:
            scales = []
            if len(self) > 0:
                h5ds.iterate(self._id, self._dimension, scales.append, 0)
            return [(self._d(h5ds.get_scale_name(x)), Dataset(x)) for x in scales]

    def keys(self):
        """ Get a list of names for the scales on this dimension. """
        with phil:
            return [key for key, _ in self.items()]

    def values(self):
        """ Get a list of Dataset for scales on this dimension. """
        with phil:
            return [val for _, val in self.items()]

    @with_phil
    def __repr__(self):
        if not self._id:
            return '<Dimension of closed HDF5 dataset>'
        return '<"%s" dimension %d of HDF5 dataset at %s>' % (self.label, self._dimension, id(self._id))