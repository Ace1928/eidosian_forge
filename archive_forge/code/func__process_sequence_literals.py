from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def _process_sequence_literals(glyphclass, kwargs, source, is_user_source):
    incompatible_literal_spec_values = []
    dataspecs = glyphclass.dataspecs()
    for var, val in kwargs.items():
        if not isinstance(val, Iterable):
            continue
        if isinstance(val, dict):
            continue
        if var not in dataspecs:
            continue
        if isinstance(val, str):
            continue
        if isinstance(dataspecs[var], ColorSpec) and dataspecs[var].is_color_tuple_shape(val):
            continue
        if isinstance(val, np.ndarray):
            if isinstance(dataspecs[var], ColorSpec):
                if val.dtype == 'uint32' and val.ndim == 1:
                    pass
                elif val.dtype == 'uint8' and val.ndim == 1:
                    pass
                elif val.dtype.kind == 'U' and val.ndim == 1:
                    pass
                elif (val.dtype == 'uint8' or val.dtype.kind == 'f') and val.ndim == 2 and (val.shape[1] in (3, 4)):
                    pass
                else:
                    raise RuntimeError(f'Color columns need to be of type uint32[N], uint8[N] or uint8/float[N, {{3, 4}}] ({var} is {val.dtype}[{', '.join(map(str, val.shape))}]')
            elif val.ndim != 1:
                raise RuntimeError(f'Columns need to be 1D ({var} is not)')
        if is_user_source:
            incompatible_literal_spec_values.append(var)
        else:
            source.add(val, name=var)
            kwargs[var] = var
    return incompatible_literal_spec_values