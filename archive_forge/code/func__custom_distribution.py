import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def _custom_distribution(self, state):
    index = state.choice(np.arange(len(self._distribution)), p=tuple(self._distribution.values()))
    return list(self._distribution.keys())[index]