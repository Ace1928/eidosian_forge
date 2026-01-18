from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
class BreakingChange(NamedTuple):
    type: BreakingChangeType
    description: str