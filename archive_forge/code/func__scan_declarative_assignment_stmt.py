from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import Decorator
from mypy.nodes import LambdaExpr
from mypy.nodes import ListExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import PlaceholderNode
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolNode
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import Type
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType
from . import apply
from . import infer
from . import names
from . import util
def _scan_declarative_assignment_stmt(cls: ClassDef, api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, attributes: List[util.SQLAlchemyAttribute]) -> None:
    """Extract mapping information from an assignment statement in a
    declarative class.

    """
    lvalue = stmt.lvalues[0]
    if not isinstance(lvalue, NameExpr):
        return
    sym = cls.info.names.get(lvalue.name)
    assert sym is not None
    node = sym.node
    if isinstance(node, PlaceholderNode):
        return
    assert node is lvalue.node
    assert isinstance(node, Var)
    if node.name == '__abstract__':
        if api.parse_bool(stmt.rvalue) is True:
            util.set_is_base(cls.info)
        return
    elif node.name == '__tablename__':
        util.set_has_table(cls.info)
    elif node.name.startswith('__'):
        return
    elif node.name == '_mypy_mapped_attrs':
        if not isinstance(stmt.rvalue, ListExpr):
            util.fail(api, '_mypy_mapped_attrs is expected to be a list', stmt)
        else:
            for item in stmt.rvalue.items:
                if isinstance(item, (NameExpr, StrExpr)):
                    apply.apply_mypy_mapped_attr(cls, api, item, attributes)
    left_hand_mapped_type: Optional[Type] = None
    left_hand_explicit_type: Optional[ProperType] = None
    if node.is_inferred or node.type is None:
        if isinstance(stmt.type, UnboundType):
            left_hand_explicit_type = stmt.type
            if stmt.type.name == 'Mapped':
                mapped_sym = api.lookup_qualified('Mapped', cls)
                if mapped_sym is not None and mapped_sym.node is not None and (names.type_id_for_named_node(mapped_sym.node) is names.MAPPED):
                    left_hand_explicit_type = get_proper_type(stmt.type.args[0])
                    left_hand_mapped_type = stmt.type
    else:
        node_type = get_proper_type(node.type)
        if isinstance(node_type, Instance) and names.type_id_for_named_node(node_type.type) is names.MAPPED:
            left_hand_explicit_type = get_proper_type(node_type.args[0])
            left_hand_mapped_type = node_type
        else:
            left_hand_explicit_type = node_type
            left_hand_mapped_type = None
    if isinstance(stmt.rvalue, TempNode) and left_hand_mapped_type is not None:
        python_type_for_type = left_hand_explicit_type
    elif isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue.callee, RefExpr):
        python_type_for_type = infer.infer_type_from_right_hand_nameexpr(api, stmt, node, left_hand_explicit_type, stmt.rvalue.callee)
        if python_type_for_type is None:
            return
    else:
        return
    assert python_type_for_type is not None
    attributes.append(util.SQLAlchemyAttribute(name=node.name, line=stmt.line, column=stmt.column, typ=python_type_for_type, info=cls.info))
    apply.apply_type_to_mapped_statement(api, stmt, lvalue, left_hand_explicit_type, python_type_for_type)