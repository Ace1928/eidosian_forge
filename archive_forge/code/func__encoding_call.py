import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def _encoding_call(self, method_name, *args, **kwargs):
    raw_method = getattr(self.raw, method_name)
    val = raw_method(*args, **kwargs)
    return val.encode(self.encoding, errors=self.errors)