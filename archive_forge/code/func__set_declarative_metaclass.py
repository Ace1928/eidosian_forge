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
def _set_declarative_metaclass(api: SemanticAnalyzerPluginInterface, target_cls: ClassDef) -> None:
    info = target_cls.info
    sym = api.lookup_fully_qualified_or_none('sqlalchemy.orm.decl_api.DeclarativeMeta')
    assert sym is not None and isinstance(sym.node, TypeInfo)
    info.declared_metaclass = info.metaclass_type = Instance(sym.node, [])