from __future__ import annotations
from collections.abc import (
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
import pandas.core.common as com
from pandas.core.frame import _shared_docs
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import cartesian_product
from pandas.core.series import Series
def _generate_marginal_results(table, data: DataFrame, values, rows, cols, aggfunc, observed: bool, margins_name: Hashable='All'):
    margin_keys: list | Index
    if len(cols) > 0:
        table_pieces = []
        margin_keys = []

        def _all_key(key):
            return (key, margins_name) + ('',) * (len(cols) - 1)
        if len(rows) > 0:
            margin = data[rows + values].groupby(rows, observed=observed).agg(aggfunc)
            cat_axis = 1
            for key, piece in table.T.groupby(level=0, observed=observed):
                piece = piece.T
                all_key = _all_key(key)
                piece = piece.copy()
                piece[all_key] = margin[key]
                table_pieces.append(piece)
                margin_keys.append(all_key)
        else:
            from pandas import DataFrame
            cat_axis = 0
            for key, piece in table.groupby(level=0, observed=observed):
                if len(cols) > 1:
                    all_key = _all_key(key)
                else:
                    all_key = margins_name
                table_pieces.append(piece)
                transformed_piece = DataFrame(piece.apply(aggfunc)).T
                if isinstance(piece.index, MultiIndex):
                    transformed_piece.index = MultiIndex.from_tuples([all_key], names=piece.index.names + [None])
                else:
                    transformed_piece.index = Index([all_key], name=piece.index.name)
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)
        if not table_pieces:
            return table
        else:
            result = concat(table_pieces, axis=cat_axis)
        if len(rows) == 0:
            return result
    else:
        result = table
        margin_keys = table.columns
    if len(cols) > 0:
        row_margin = data[cols + values].groupby(cols, observed=observed).agg(aggfunc)
        row_margin = row_margin.stack(future_stack=True)
        new_order_indices = [len(cols)] + list(range(len(cols)))
        new_order_names = [row_margin.index.names[i] for i in new_order_indices]
        row_margin.index = row_margin.index.reorder_levels(new_order_names)
    else:
        row_margin = data._constructor_sliced(np.nan, index=result.columns)
    return (result, margin_keys, row_margin)