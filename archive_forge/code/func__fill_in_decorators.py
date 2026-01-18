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
def _fill_in_decorators(ctx: ClassDefContext) -> None:
    for decorator in ctx.cls.decorators:
        if isinstance(decorator, nodes.CallExpr) and isinstance(decorator.callee, nodes.MemberExpr) and (decorator.callee.name == 'as_declarative_base'):
            target = decorator.callee
        elif isinstance(decorator, nodes.MemberExpr) and decorator.name == 'mapped':
            target = decorator
        else:
            continue
        if isinstance(target.expr, NameExpr):
            sym = ctx.api.lookup_qualified(target.expr.name, target, suppress_errors=True)
        else:
            continue
        if sym and sym.node:
            sym_type = get_proper_type(sym.type)
            if isinstance(sym_type, Instance):
                target.fullname = f'{sym_type.type.fullname}.{target.name}'
            else:
                util.fail(ctx.api, "Class decorator called %s(), but we can't tell if it's from an ORM registry.  Please annotate the registry assignment, e.g. my_registry: registry = registry()" % target.name, sym.node)