from __future__ import annotations
from typing import Optional
from typing import Sequence
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import LambdaExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.subtypes import is_subtype
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnionType
from . import names
from . import util
def _infer_type_from_mapped(api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, node: Var, left_hand_explicit_type: Optional[ProperType], infer_from_right_side: RefExpr) -> Optional[ProperType]:
    """Infer the type of mapping from a right side expression
    that returns Mapped.


    """
    assert isinstance(stmt.rvalue, CallExpr)
    the_mapped_type = util.type_for_callee(infer_from_right_side)
    return infer_type_from_left_hand_type_only(api, node, left_hand_explicit_type)