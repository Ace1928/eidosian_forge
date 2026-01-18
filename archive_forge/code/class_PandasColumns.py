from functools import partial
import pickle
import numpy as np
import pandas as pd
from pandas.core.internals import create_block_manager_from_blocks, make_block
from . import numpy as pnp
from .core import Interface
from .encode import Encode
from .utils import extend, framesplit, frame
class PandasColumns(Interface):

    def __init__(self, partd=None):
        self.partd = pnp.Numpy(partd)
        Interface.__init__(self)

    def append(self, data, **kwargs):
        for k, df in data.items():
            self.iset(extend(k, '.columns'), dumps(list(df.columns)))
            self.iset(extend(k, '.index-name'), dumps(df.index.name))
        arrays = {extend(k, col): df[col].values for k, df in data.items() for col in df.columns}
        arrays.update({extend(k, '.index'): df.index.values for k, df in data.items()})
        self.partd.append(arrays, **kwargs)

    def _get(self, keys, columns=None, **kwargs):
        if columns is None:
            columns = self.partd.partd.get([extend(k, '.columns') for k in keys], **kwargs)
            columns = list(map(pickle.loads, columns))
        else:
            columns = [columns] * len(keys)
        index_names = self.partd.partd.get([extend(k, '.index-name') for k in keys], **kwargs)
        index_names = map(pickle.loads, index_names)
        keys = [[extend(k, '.index'), [extend(k, col) for col in cols]] for k, cols in zip(keys, columns)]
        arrays = self.partd.get(keys, **kwargs)
        return [pd.DataFrame(dict(zip(cols, arrs)), columns=cols, index=pd.Index(index, name=iname)) for iname, (index, arrs), cols in zip(index_names, arrays, columns)]

    def __getstate__(self):
        return {'partd': self.partd}

    def _iset(self, key, value):
        return self.partd._iset(key, value)

    def drop(self):
        return self.partd.drop()

    @property
    def lock(self):
        return self.partd.partd.lock

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(self, *args)

    def __del__(self):
        self.partd.__del__()