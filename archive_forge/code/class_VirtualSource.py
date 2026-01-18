from copy import deepcopy as copy
from collections import namedtuple
import numpy as np
from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t
class VirtualSource:
    """Source definition for virtual data sets.

    Instantiate this class to represent an entire source dataset, and then
    slice it to indicate which regions should be used in the virtual dataset.

    path_or_dataset
        The path to a file, or an h5py dataset. If a dataset is given,
        no other parameters are allowed, as the relevant values are taken from
        the dataset instead.
    name
        The name of the source dataset within the file.
    shape
        A tuple giving the shape of the dataset.
    dtype
        Numpy dtype or string.
    maxshape
        The source dataset is resizable up to this shape. Use None for
        axes you want to be unlimited.
    """

    def __init__(self, path_or_dataset, name=None, shape=None, dtype=None, maxshape=None):
        from .dataset import Dataset
        if isinstance(path_or_dataset, Dataset):
            failed = {k: v for k, v in {'name': name, 'shape': shape, 'dtype': dtype, 'maxshape': maxshape}.items() if v is not None}
            if failed:
                raise TypeError('If a Dataset is passed as the first argument then no other arguments may be passed.  You passed {failed}'.format(failed=failed))
            ds = path_or_dataset
            path = ds.file.filename
            name = ds.name
            shape = ds.shape
            dtype = ds.dtype
            maxshape = ds.maxshape
        else:
            path = path_or_dataset
            if name is None:
                raise TypeError('The name parameter is required when specifying a source by path')
            if shape is None:
                raise TypeError('The shape parameter is required when specifying a source by path')
            elif isinstance(shape, int):
                shape = (shape,)
            if isinstance(maxshape, int):
                maxshape = (maxshape,)
        self.path = path
        self.name = name
        self.dtype = dtype
        if maxshape is None:
            self.maxshape = shape
        else:
            self.maxshape = tuple([h5s.UNLIMITED if ix is None else ix for ix in maxshape])
        self.sel = SimpleSelection(shape)
        self._all_selected = True

    @property
    def shape(self):
        return self.sel.array_shape

    def __getitem__(self, key):
        if not self._all_selected:
            raise RuntimeError('VirtualSource objects can only be sliced once.')
        tmp = copy(self)
        tmp.sel = select(self.shape, key, dataset=None)
        _convert_space_for_key(tmp.sel.id, key)
        tmp._all_selected = False
        return tmp