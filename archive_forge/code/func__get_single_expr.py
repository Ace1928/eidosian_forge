from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Set, Union, Tuple
from triad.collections import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.schema import unquote_name
def _get_single_expr(self, expr: str) -> Union[Column, 'DataFrame']:
    ee = expr.split('.', 1)
    if len(ee) == 1:
        df_name, col_name = ('', unquote_name(expr))
    else:
        df_name, col_name = (ee[0].strip(), unquote_name(ee[1]))
    if col_name == '*':
        if df_name != '':
            return self[df_name][col_name]
        else:
            assert_or_throw(not self._has_overlap, ValueError('There is schema overlap'))
            return DataFrame(*list(self.values()))
    elif df_name == '':
        dfs = self._col_to_df[col_name]
        assert_or_throw(len(dfs) > 0, ValueError(f'{col_name} is not defined'))
        assert_or_throw(len(dfs) == 1, ValueError(f'{col_name} in these dataframes {dfs}'))
        return self[next(iter(dfs))][col_name]
    else:
        df = self[df_name]
        assert_or_throw(col_name in df, ValueError(f'{df_name} does not have {col_name}'))
        return df[col_name]