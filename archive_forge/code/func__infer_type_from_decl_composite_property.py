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
def _infer_type_from_decl_composite_property(api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, node: Var, left_hand_explicit_type: Optional[ProperType]) -> Optional[ProperType]:
    """Infer the type of mapping from a Composite."""
    assert isinstance(stmt.rvalue, CallExpr)
    target_cls_arg = stmt.rvalue.args[0]
    python_type_for_type = None
    if isinstance(target_cls_arg, NameExpr) and isinstance(target_cls_arg.node, TypeInfo):
        related_object_type = target_cls_arg.node
        python_type_for_type = Instance(related_object_type, [])
    else:
        python_type_for_type = None
    if python_type_for_type is None:
        return infer_type_from_left_hand_type_only(api, node, left_hand_explicit_type)
    elif left_hand_explicit_type is not None:
        return _infer_type_from_left_and_inferred_right(api, node, left_hand_explicit_type, python_type_for_type)
    else:
        return python_type_for_type