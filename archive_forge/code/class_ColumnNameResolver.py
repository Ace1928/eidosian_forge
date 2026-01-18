from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Set, Union, Tuple
from triad.collections import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.schema import unquote_name
class ColumnNameResolver(object):

    def __init__(self, *args: Any, **kwargs: Any):
        self._data: Dict[str, List[str]] = OrderedDict()
        for d in args:
            if isinstance(d, Dict):
                self._update(d)
            else:
                raise ValueError(f'{d} is not valid to initialize DataFrames')
        self._update(kwargs)
        self._col_to_df: Dict[str, Set[str]] = defaultdict(set)
        for k, v in self._data.items():
            for c in v:
                self._col_to_df[c].add(k)
        self._has_overlap = any((len(v) > 1 for v in self._col_to_df.values()))

    @property
    def has_overlap(self):
        return self._has_overlap

    def get_col(self, expr) -> Tuple[str, str]:
        assert_or_throw('*' not in expr, ValueError(f'{expr} is invalid to get a single column'))
        cols = self._get_cols(expr)
        assert_or_throw(len(cols) == 1, f'expected single column but get [{cols}]')
        return cols[0]

    def get_cols(self, *exprs: Any, ensure_distinct: bool=False, ensure_single_df: bool=False) -> List[Tuple[str, str]]:
        res: List[Tuple[str, str]] = []
        for e in exprs:
            res += self._get_cols(e)
        if ensure_distinct:
            assert_or_throw(len(set(res)) == len(res), f'there are duplicates {res}')
        if ensure_single_df:
            assert_or_throw(len(set((x[0] for x in res))) == 1, f'not from single dataframe {res}')
        return res

    def _update(self, dfs: Dict):
        for k, v in dfs.items():
            assert_or_throw(isinstance(k, str), ValueError(f'{k} is not string'))
            assert_or_throw(k != '', ValueError("key can't be empty"))
            if isinstance(v, DataFrame):
                self._data[k] = list(v.keys())
            elif isinstance(v, Iterable):
                self._data[k] = list(v)
            else:
                raise ValueError(f'{v} is invalid')

    def _get_cols(self, expr: str) -> List[Tuple[str, str]]:
        ee = expr.split('.', 1)
        if len(ee) == 1:
            df_name, col_name = ('', unquote_name(expr))
        else:
            df_name, col_name = (ee[0], unquote_name(ee[1]))
        if col_name == '*':
            if df_name != '':
                return [(df_name, x) for x in self._data[df_name]]
            else:
                assert_or_throw(not self.has_overlap, ValueError('There is schema overlap'))
                res: List[Tuple[str, str]] = []
                for k, v in self._data.items():
                    res += [(k, x) for x in v]
                return res
        elif df_name == '':
            dfs = self._col_to_df[col_name]
            assert_or_throw(len(dfs) > 0, ValueError(f'{col_name} is not defined'))
            assert_or_throw(len(dfs) == 1, ValueError(f'{col_name} in these dataframes {dfs}'))
            return [(next(iter(dfs)), col_name)]
        else:
            assert_or_throw(col_name in self._col_to_df and df_name in self._col_to_df[col_name], ValueError(f'{expr} does not exist'))
            return [(df_name, col_name)]