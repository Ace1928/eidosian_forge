from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Tuple, Union
from antlr4.tree.Tree import TerminalNode, Token, Tree
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from _qpd_antlr import QPDParser as qp
from qpd._parser.sql import QPDSql
from triad.utils.schema import unquote_name
from qpd.constants import AGGREGATION_FUNCTIONS, JOIN_TYPES
from qpd.dataframe import DataFrame, DataFrames
from qpd.workflow import QPDWorkflow
def add_column_mentions(self, ctx: Any) -> ColumnMentions:
    if 'col_mentions' in self._data and id(ctx) in self._data['col_mentions']:
        return self.get_column_mentions(ctx)
    expr = self.to_str(ctx, '', unquote=False)
    assert_or_throw(expr != '*', ValueError("'*' shouldn't be used here"))
    mentions = ColumnMentions(*self.dfs._resolver.get_cols(expr))
    if 'col_mentions' not in self._data:
        self._data['col_mentions'] = {id(ctx): mentions}
    else:
        self._data['col_mentions'][id(ctx)] = mentions
    if 'all_mentions' not in self._data:
        self._data['all_mentions'] = ColumnMentions()
    self._data['all_mentions'].add(mentions)
    return mentions