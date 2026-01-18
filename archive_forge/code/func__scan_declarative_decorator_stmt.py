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
def _scan_declarative_decorator_stmt(cls: ClassDef, api: SemanticAnalyzerPluginInterface, stmt: Decorator, attributes: List[util.SQLAlchemyAttribute]) -> None:
    """Extract mapping information from a @declared_attr in a declarative
    class.

    E.g.::

        @reg.mapped
        class MyClass:
            # ...

            @declared_attr
            def updated_at(cls) -> Column[DateTime]:
                return Column(DateTime)

    Will resolve in mypy as::

        @reg.mapped
        class MyClass:
            # ...

            updated_at: Mapped[Optional[datetime.datetime]]

    """
    for dec in stmt.decorators:
        if isinstance(dec, (NameExpr, MemberExpr, SymbolNode)) and names.type_id_for_named_node(dec) is names.DECLARED_ATTR:
            break
    else:
        return
    dec_index = cls.defs.body.index(stmt)
    left_hand_explicit_type: Optional[ProperType] = None
    if util.name_is_dunder(stmt.name):
        any_ = AnyType(TypeOfAny.special_form)
        left_node = NameExpr(stmt.var.name)
        left_node.node = stmt.var
        new_stmt = AssignmentStmt([left_node], TempNode(any_))
        new_stmt.type = left_node.node.type
        cls.defs.body[dec_index] = new_stmt
        return
    elif isinstance(stmt.func.type, CallableType):
        func_type = stmt.func.type.ret_type
        if isinstance(func_type, UnboundType):
            type_id = names.type_id_for_unbound_type(func_type, cls, api)
        else:
            return
        if type_id in {names.MAPPED, names.RELATIONSHIP, names.COMPOSITE_PROPERTY, names.MAPPER_PROPERTY, names.SYNONYM_PROPERTY, names.COLUMN_PROPERTY} and func_type.args:
            left_hand_explicit_type = get_proper_type(func_type.args[0])
        elif type_id is names.COLUMN and func_type.args:
            typeengine_arg = func_type.args[0]
            if isinstance(typeengine_arg, UnboundType):
                sym = api.lookup_qualified(typeengine_arg.name, typeengine_arg)
                if sym is not None and isinstance(sym.node, TypeInfo):
                    if names.has_base_type_id(sym.node, names.TYPEENGINE):
                        left_hand_explicit_type = UnionType([infer.extract_python_type_from_typeengine(api, sym.node, []), NoneType()])
                    else:
                        util.fail(api, "Column type should be a TypeEngine subclass not '{}'".format(sym.node.fullname), func_type)
    if left_hand_explicit_type is None:
        msg = "Can't infer type from @declared_attr on function '{}';  please specify a return type from this function that is one of: Mapped[<python type>], relationship[<target class>], Column[<TypeEngine>], MapperProperty[<python type>]"
        util.fail(api, msg.format(stmt.var.name), stmt)
        left_hand_explicit_type = AnyType(TypeOfAny.special_form)
    left_node = NameExpr(stmt.var.name)
    left_node.node = stmt.var
    if isinstance(left_hand_explicit_type, UnboundType):
        left_hand_explicit_type = get_proper_type(util.unbound_to_instance(api, left_hand_explicit_type))
    left_node.node.type = api.named_type(names.NAMED_TYPE_SQLA_MAPPED, [left_hand_explicit_type])
    rvalue = names.expr_to_mapped_constructor(LambdaExpr(stmt.func.arguments, stmt.func.body))
    new_stmt = AssignmentStmt([left_node], rvalue)
    new_stmt.type = left_node.node.type
    attributes.append(util.SQLAlchemyAttribute(name=left_node.name, line=stmt.line, column=stmt.column, typ=left_hand_explicit_type, info=cls.info))
    cls.defs.body[dec_index] = new_stmt