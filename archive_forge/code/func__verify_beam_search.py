import copy
from enum import IntEnum
from functools import cached_property
from typing import Callable, List, Optional, Union
import torch
def _verify_beam_search(self) -> None:
    if self.best_of == 1:
        raise ValueError(f'best_of must be greater than 1 when using beam search. Got {self.best_of}.')
    if self.temperature > _SAMPLING_EPS:
        raise ValueError('temperature must be 0 when using beam search.')
    if self.top_p < 1.0 - _SAMPLING_EPS:
        raise ValueError('top_p must be 1 when using beam search.')
    if self.top_k != -1:
        raise ValueError('top_k must be -1 when using beam search.')
    if self.early_stopping not in [True, False, 'never']:
        raise ValueError(f"early_stopping must be True, False, or 'never', got {self.early_stopping}.")