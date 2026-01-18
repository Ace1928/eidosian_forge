import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def _loguniform(self, state):
    output = int(2 ** state.uniform(low=np.log2(self._minval) if self._minval is not None else None, high=np.log2(self._maxval) if self._maxval is not None else None))
    if self._minval is not None and output < self._minval:
        return self._minval
    if self._maxval is not None and output > self._maxval:
        return self._maxval
    return output