import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def _get_batch(self, sequence, indices):
    if isinstance(sequence, list):
        subseq = [sequence[i] for i in indices]
    elif isinstance(sequence, tuple):
        subseq = tuple((sequence[i] for i in indices))
    else:
        subseq = sequence[indices]
    if is_xp_array(subseq):
        subseq = self.as_contig(self.xp.asarray(subseq))
    return subseq