from statsmodels.compat.python import lzip
import numpy as np
from statsmodels.tools.testing import Holder
class HolderTuple(Holder):
    """Holder class with indexing

    """

    def __init__(self, tuple_=None, **kwds):
        super().__init__(**kwds)
        if tuple_ is not None:
            self.tuple = tuple((getattr(self, att) for att in tuple_))
        else:
            self.tuple = (self.statistic, self.pvalue)

    def __iter__(self):
        yield from self.tuple

    def __getitem__(self, idx):
        return self.tuple[idx]

    def __len__(self):
        return len(self.tuple)

    def __array__(self, dtype=None, copy=True):
        return np.array(list(self.tuple), dtype=dtype, copy=copy)