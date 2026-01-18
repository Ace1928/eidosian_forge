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
class ColumnMentions(object):

    def __init__(self, *data: Any):
        self._data: Dict[ColumnMention, int] = OrderedDict()
        self.add(*data)

    def add(self, *data: Any) -> 'ColumnMentions':
        for x in data:
            if isinstance(x, ColumnMention):
                self._data[x] = 1
            elif isinstance(x, ColumnMentions):
                self._data.update(x._data)
            elif isinstance(x, tuple):
                self._data[ColumnMention(x[0], x[1])] = 1
            else:
                raise ValueError(f'{x} is invalid')
        return self

    def __contains__(self, m: ColumnMention) -> bool:
        return m in self._data

    def __iter__(self) -> Iterator[ColumnMention]:
        return iter(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return ','.join((x.__repr__() for x in self))

    @property
    def single(self) -> ColumnMention:
        assert_or_throw(len(self._data) == 1, f'multiple mentions: {self._data}')
        return next(iter(self._data.keys()))