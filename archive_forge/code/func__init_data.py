from collections import OrderedDict
import numpy as np
from ..ndarray.sparse import CSRNDArray
from ..ndarray.sparse import array as sparse_array
from ..ndarray import NDArray
from ..ndarray import array
def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert data is not None or allow_empty
    if data is None:
        data = []
    if isinstance(data, (np.ndarray, NDArray, h5py.Dataset) if h5py else (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert len(data) > 0
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])
        else:
            data = OrderedDict([('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError('Input must be NDArray, numpy.ndarray, h5py.Dataset ' + 'a list of them or dict with them as values')
    for k, v in data.items():
        if not isinstance(v, (NDArray, h5py.Dataset) if h5py else NDArray):
            try:
                data[k] = array(v)
            except:
                raise TypeError("Invalid type '%s' for %s, " % (type(v), k) + 'should be NDArray, numpy.ndarray or h5py.Dataset')
    return list(sorted(data.items()))