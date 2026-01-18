import itertools
from typing import Dict, List
from mypy.checker import TypeChecker
from mypy.errorcodes import ErrorCode
from mypy.errors import Errors
from mypy.nodes import CallExpr, Decorator, Expression, FuncDef, MypyFile, NameExpr
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, FunctionContext, Plugin
from mypy.subtypes import is_subtype
from mypy.types import CallableType, Instance, Type, TypeVarType
class IntrospectChecker(TypeChecker):

    def __init__(self, errors: Errors, modules: Dict[str, MypyFile], options: Options, tree: MypyFile, path: str, plugin: Plugin, per_line_checking_time_ns: Dict[int, int]):
        self._error_messages: List[str] = []
        super().__init__(errors, modules, options, tree, path, plugin, per_line_checking_time_ns)