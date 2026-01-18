import os
import re
import functools
import itertools
import warnings
import weakref
import contextlib
import operator
from operator import itemgetter, index as opindex, methodcaller
from collections.abc import Mapping
import numpy as np
from . import format
from ._datasource import DataSource
from numpy.core import overrides
from numpy.core.multiarray import packbits, unpackbits
from numpy.core._multiarray_umath import _load_from_filelike
from numpy.core.overrides import set_array_function_like_doc, set_module
from ._iotools import (
from numpy.compat import (
def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    import zipfile
    if not hasattr(file, 'write'):
        file = os_fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'
    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError('Cannot use un-named variables and keyword %s' % key)
        namedict[key] = val
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED
    zipf = zipfile_factory(file, mode='w', compression=compression)
    for key, val in namedict.items():
        fname = key + '.npy'
        val = np.asanyarray(val)
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            format.write_array(fid, val, allow_pickle=allow_pickle, pickle_kwargs=pickle_kwargs)
    zipf.close()