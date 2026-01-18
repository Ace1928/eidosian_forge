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
def add_additional_orm_attributes(cls: ClassDef, api: SemanticAnalyzerPluginInterface, attributes: List[util.SQLAlchemyAttribute]) -> None:
    """Apply __init__, __table__ and other attributes to the mapped class."""
    info = util.info_for_cls(cls, api)
    if info is None:
        return
    is_base = util.get_is_base(info)
    if '__init__' not in info.names and (not is_base):
        mapped_attr_names = {attr.name: attr.type for attr in attributes}
        for base in info.mro[1:-1]:
            if 'sqlalchemy' not in info.metadata:
                continue
            base_cls_attributes = util.get_mapped_attributes(base, api)
            if base_cls_attributes is None:
                continue
            for attr in base_cls_attributes:
                mapped_attr_names.setdefault(attr.name, attr.type)
        arguments = []
        for name, typ in mapped_attr_names.items():
            if typ is None:
                typ = AnyType(TypeOfAny.special_form)
            arguments.append(Argument(variable=Var(name, typ), type_annotation=typ, initializer=TempNode(typ), kind=ARG_NAMED_OPT))
        add_method_to_class(api, cls, '__init__', arguments, NoneTyp())
    if '__table__' not in info.names and util.get_has_table(info):
        _apply_placeholder_attr_to_class(api, cls, 'sqlalchemy.sql.schema.Table', '__table__')
    if not is_base:
        _apply_placeholder_attr_to_class(api, cls, 'sqlalchemy.orm.mapper.Mapper', '__mapper__')