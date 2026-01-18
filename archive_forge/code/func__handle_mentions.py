from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def _handle_mentions(self) -> None:
    for m in self.get('non_agg_mentions'):
        func = AggFunctionSpec('first', unique=False, dropna=False)
        self._gp_map[m.encoded] = (m.encoded, func)