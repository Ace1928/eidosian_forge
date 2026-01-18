from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from mypy.nodes import ARG_NAMED_OPT
from mypy.nodes import Argument
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import MDEF
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneTyp
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType
from . import infer
from . import util
from .names import expr_to_mapped_constructor
from .names import NAMED_TYPE_SQLA_MAPPED
def _apply_placeholder_attr_to_class(api: SemanticAnalyzerPluginInterface, cls: ClassDef, qualified_name: str, attrname: str) -> None:
    sym = api.lookup_fully_qualified_or_none(qualified_name)
    if sym:
        assert isinstance(sym.node, TypeInfo)
        type_: ProperType = Instance(sym.node, [])
    else:
        type_ = AnyType(TypeOfAny.special_form)
    var = Var(attrname)
    var._fullname = cls.fullname + '.' + attrname
    var.info = cls.info
    var.type = type_
    cls.info.names[attrname] = SymbolTableNode(MDEF, var)