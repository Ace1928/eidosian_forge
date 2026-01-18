from __future__ import annotations
import sys
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.helper import flatten
def _calc_start_end(self, start: int, end: int) -> t.Dict[str, t.Optional[t.Union[str, exp.Expression]]]:
    kwargs: t.Dict[str, t.Optional[t.Union[str, exp.Expression]]] = {'start_side': None, 'end_side': None}
    if start == Window.currentRow:
        kwargs['start'] = 'CURRENT ROW'
    else:
        kwargs = {**kwargs, **{'start_side': 'PRECEDING', 'start': 'UNBOUNDED' if start <= Window.unboundedPreceding else F.lit(start).expression}}
    if end == Window.currentRow:
        kwargs['end'] = 'CURRENT ROW'
    else:
        kwargs = {**kwargs, **{'end_side': 'FOLLOWING', 'end': 'UNBOUNDED' if end >= Window.unboundedFollowing else F.lit(end).expression}}
    return kwargs