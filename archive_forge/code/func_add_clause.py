from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
def add_clause(self, clause):
    assert isinstance(clause, _LandingPadClause)
    self.clauses.append(clause)