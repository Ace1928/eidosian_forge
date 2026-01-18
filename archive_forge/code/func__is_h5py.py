from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
def _is_h5py(obj, allow_file=True, allow_group=True, allow_dataset=True):
    if not utils._try_import('h5py'):
        return False
    else:
        types = []
        if allow_file:
            types.append(h5py.File)
        if allow_group:
            types.append(h5py.Group)
        if allow_dataset:
            types.append(h5py.Dataset)
        return isinstance(obj, tuple(types))