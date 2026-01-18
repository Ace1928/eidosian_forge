from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type as TypingType
from typing import Union
from mypy import nodes
from mypy.mro import calculate_mro
from mypy.mro import MroError
from mypy.nodes import Block
from mypy.nodes import ClassDef
from mypy.nodes import GDEF
from mypy.nodes import MypyFile
from mypy.nodes import NameExpr
from mypy.nodes import SymbolTable
from mypy.nodes import SymbolTableNode
from mypy.nodes import TypeInfo
from mypy.plugin import AttributeContext
from mypy.plugin import ClassDefContext
from mypy.plugin import DynamicClassDefContext
from mypy.plugin import Plugin
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import Type
from . import decl_class
from . import names
from . import util
class SQLAlchemyPlugin(Plugin):

    def get_dynamic_class_hook(self, fullname: str) -> Optional[Callable[[DynamicClassDefContext], None]]:
        if names.type_id_for_fullname(fullname) is names.DECLARATIVE_BASE:
            return _dynamic_class_hook
        return None

    def get_customize_class_mro_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        return _fill_in_decorators

    def get_class_decorator_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        sym = self.lookup_fully_qualified(fullname)
        if sym is not None and sym.node is not None:
            type_id = names.type_id_for_named_node(sym.node)
            if type_id is names.MAPPED_DECORATOR:
                return _cls_decorator_hook
            elif type_id in (names.AS_DECLARATIVE, names.AS_DECLARATIVE_BASE):
                return _base_cls_decorator_hook
            elif type_id is names.DECLARATIVE_MIXIN:
                return _declarative_mixin_hook
        return None

    def get_metaclass_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        if names.type_id_for_fullname(fullname) is names.DECLARATIVE_META:
            return _metaclass_cls_hook
        return None

    def get_base_class_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo) and util.has_declarative_base(sym.node):
            return _base_cls_hook
        return None

    def get_attribute_hook(self, fullname: str) -> Optional[Callable[[AttributeContext], Type]]:
        if fullname.startswith('sqlalchemy.orm.attributes.QueryableAttribute.'):
            return _queryable_getattr_hook
        return None

    def get_additional_deps(self, file: MypyFile) -> List[Tuple[int, str, int]]:
        return [(10, 'sqlalchemy.orm', -1), (10, 'sqlalchemy.orm.attributes', -1), (10, 'sqlalchemy.orm.decl_api', -1)]