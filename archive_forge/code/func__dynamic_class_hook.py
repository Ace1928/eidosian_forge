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
def _dynamic_class_hook(ctx: DynamicClassDefContext) -> None:
    """Generate a declarative Base class when the declarative_base() function
    is encountered."""
    _add_globals(ctx)
    cls = ClassDef(ctx.name, Block([]))
    cls.fullname = ctx.api.qualified_name(ctx.name)
    info = TypeInfo(SymbolTable(), cls, ctx.api.cur_mod_id)
    cls.info = info
    _set_declarative_metaclass(ctx.api, cls)
    cls_arg = util.get_callexpr_kwarg(ctx.call, 'cls', expr_types=(NameExpr,))
    if cls_arg is not None and isinstance(cls_arg.node, TypeInfo):
        util.set_is_base(cls_arg.node)
        decl_class.scan_declarative_assignments_and_apply_types(cls_arg.node.defn, ctx.api, is_mixin_scan=True)
        info.bases = [Instance(cls_arg.node, [])]
    else:
        obj = ctx.api.named_type(names.NAMED_TYPE_BUILTINS_OBJECT)
        info.bases = [obj]
    try:
        calculate_mro(info)
    except MroError:
        util.fail(ctx.api, 'Not able to calculate MRO for declarative base', ctx.call)
        obj = ctx.api.named_type(names.NAMED_TYPE_BUILTINS_OBJECT)
        info.bases = [obj]
        info.fallback_to_any = True
    ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, info))
    util.set_is_base(info)