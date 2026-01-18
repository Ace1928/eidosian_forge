from __future__ import annotations
import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import remove_missing
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from .guide import GuideElements, guide
def _calculate_rows_and_cols(self, elements: GuideElementsLegend) -> TupleInt2:
    nrow, ncol = (self.nrow, self.ncol)
    nbreak = len(self.key)
    if nrow and ncol:
        if nrow * ncol < nbreak:
            raise PlotnineError('nrow x ncol needs to be larger than the number of breaks')
        return (nrow, ncol)
    if (nrow, ncol) == (None, None):
        if elements.is_horizontal:
            nrow = int(np.ceil(nbreak / 5))
        else:
            ncol = int(np.ceil(nbreak / 20))
    if nrow is None:
        ncol = cast(int, ncol)
        nrow = int(np.ceil(nbreak / ncol))
    elif ncol is None:
        nrow = cast(int, nrow)
        ncol = int(np.ceil(nbreak / nrow))
    return (nrow, ncol)