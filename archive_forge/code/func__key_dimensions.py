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
@cached_property
def _key_dimensions(self) -> list[TupleFloat2]:
    """
        key width and key height for each legend entry

        Take a peak into data['size'] to make sure the legend key
        dimensions are big enough.
        """
    guide = cast(guide_legend, self.guide)
    min_size = (self.theme.getp('legend_key_width'), self.theme.getp('legend_key_height'))
    sizes: list[list[TupleFloat2]] = []
    for params in guide._layer_parameters:
        sizes.append([])
        get_key_size = params.geom.legend_key_size
        for i in range(len(params.data)):
            key_data = params.data.iloc[i]
            sizes[-1].append(get_key_size(key_data, min_size, params.layer))
    arr = np.max(sizes, axis=0)
    return [tuple(row) for row in arr]