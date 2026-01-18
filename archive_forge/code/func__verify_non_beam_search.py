import copy
from enum import IntEnum
from functools import cached_property
from typing import Callable, List, Optional, Union
import torch
def _verify_non_beam_search(self) -> None:
    if self.early_stopping is not False:
        raise ValueError('early_stopping is not effective and must be False when not using beam search.')
    if self.length_penalty < 1.0 - _SAMPLING_EPS or self.length_penalty > 1.0 + _SAMPLING_EPS:
        raise ValueError('length_penalty is not effective and must be the default value of 1.0 when not using beam search.')