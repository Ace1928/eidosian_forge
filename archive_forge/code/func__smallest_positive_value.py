from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
@cached_property
def _smallest_positive_value(self) -> float:
    """Return the smallest positive value representable by the probs dtype.
        This value is used when constructing a distribution from which to sample
        recovered tokens in the first rejection case.

        See _get_recovered_probs for more details

        Note that this isn't actually the smallest positive value representable
        by float32, but the smallest positive normal value.
        See https://en.wikipedia.org/wiki/Subnormal_number for more information.
        """
    return torch.finfo(self.probs_dtype).tiny