from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, cast
from ..error import GraphQLError
from ..language import (
from ..type import (
from ..utilities import TypeInfo, TypeInfoVisitor
class VariableUsageVisitor(Visitor):
    """Visitor adding all variable usages to a given list."""
    usages: List[VariableUsage]

    def __init__(self, type_info: TypeInfo):
        super().__init__()
        self.usages = []
        self._append_usage = self.usages.append
        self._type_info = type_info

    def enter_variable_definition(self, *_args: Any) -> VisitorAction:
        return self.SKIP

    def enter_variable(self, node: VariableNode, *_args: Any) -> VisitorAction:
        type_info = self._type_info
        usage = VariableUsage(node, type_info.get_input_type(), type_info.get_default_value())
        self._append_usage(usage)
        return None