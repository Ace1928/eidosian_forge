from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def _handle_window_funcs(self) -> None:
    for f_ctx in self.get('window_funcs', None):
        func, args = self.visit(f_ctx)
        name = self._obj_to_col_name(self.expr(f_ctx))
        args = self._get_func_args(args)
        self._windows[name] = (args, func)
        self._f_to_name[id(f_ctx)] = name