import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def _stoichs(self, attr, keys=None):
    import numpy as np
    if keys is None:
        keys = self.substances.keys()
    return np.array([getattr(eq, attr)(keys) for eq in self.rxns], dtype=object)