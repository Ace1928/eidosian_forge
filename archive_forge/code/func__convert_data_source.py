from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def _convert_data_source(kwargs):
    is_user_source = kwargs.get('source', None) is not None
    if is_user_source:
        source = kwargs['source']
        if not isinstance(source, ColumnarDataSource):
            try:
                source = ColumnDataSource(source)
            except ValueError as err:
                msg = f'Failed to auto-convert {type(source)} to ColumnDataSource.\n Original error: {err}'
                raise ValueError(msg).with_traceback(sys.exc_info()[2])
            kwargs['source'] = source
    return is_user_source