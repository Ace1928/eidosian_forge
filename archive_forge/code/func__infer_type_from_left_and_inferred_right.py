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
def _infer_type_from_left_and_inferred_right(api: SemanticAnalyzerPluginInterface, node: Var, left_hand_explicit_type: ProperType, python_type_for_type: ProperType, orig_left_hand_type: Optional[ProperType]=None, orig_python_type_for_type: Optional[ProperType]=None) -> Optional[ProperType]:
    """Validate type when a left hand annotation is present and we also
    could infer the right hand side::

        attrname: SomeType = Column(SomeDBType)

    """
    if orig_left_hand_type is None:
        orig_left_hand_type = left_hand_explicit_type
    if orig_python_type_for_type is None:
        orig_python_type_for_type = python_type_for_type
    if not is_subtype(left_hand_explicit_type, python_type_for_type):
        effective_type = api.named_type(names.NAMED_TYPE_SQLA_MAPPED, [orig_python_type_for_type])
        msg = "Left hand assignment '{}: {}' not compatible with ORM mapped expression of type {}"
        util.fail(api, msg.format(node.name, util.format_type(orig_left_hand_type, api.options), util.format_type(effective_type, api.options)), node)
    return orig_left_hand_type