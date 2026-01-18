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
def _scan_symbol_table_entry(cls: ClassDef, api: SemanticAnalyzerPluginInterface, name: str, value: SymbolTableNode, attributes: List[util.SQLAlchemyAttribute]) -> None:
    """Extract mapping information from a SymbolTableNode that's in the
    type.names dictionary.

    """
    value_type = get_proper_type(value.type)
    if not isinstance(value_type, Instance):
        return
    left_hand_explicit_type = None
    type_id = names.type_id_for_named_node(value_type.type)
    err = False
    if type_id in {names.MAPPED, names.RELATIONSHIP, names.COMPOSITE_PROPERTY, names.MAPPER_PROPERTY, names.SYNONYM_PROPERTY, names.COLUMN_PROPERTY}:
        if value_type.args:
            left_hand_explicit_type = get_proper_type(value_type.args[0])
        else:
            err = True
    elif type_id is names.COLUMN:
        if not value_type.args:
            err = True
        else:
            typeengine_arg: Union[ProperType, TypeInfo] = get_proper_type(value_type.args[0])
            if isinstance(typeengine_arg, Instance):
                typeengine_arg = typeengine_arg.type
            if isinstance(typeengine_arg, (UnboundType, TypeInfo)):
                sym = api.lookup_qualified(typeengine_arg.name, typeengine_arg)
                if sym is not None and isinstance(sym.node, TypeInfo):
                    if names.has_base_type_id(sym.node, names.TYPEENGINE):
                        left_hand_explicit_type = UnionType([infer.extract_python_type_from_typeengine(api, sym.node, []), NoneType()])
                    else:
                        util.fail(api, "Column type should be a TypeEngine subclass not '{}'".format(sym.node.fullname), value_type)
    if err:
        msg = "Can't infer type from attribute {} on class {}. please specify a return type from this function that is one of: Mapped[<python type>], relationship[<target class>], Column[<TypeEngine>], MapperProperty[<python type>]"
        util.fail(api, msg.format(name, cls.name), cls)
        left_hand_explicit_type = AnyType(TypeOfAny.special_form)
    if left_hand_explicit_type is not None:
        assert value.node is not None
        attributes.append(util.SQLAlchemyAttribute(name=name, line=value.node.line, column=value.node.column, typ=left_hand_explicit_type, info=cls.info))