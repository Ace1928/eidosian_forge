from copy import deepcopy
import sys
from types import ModuleType
from typing import Callable, Collection, Dict, Optional, TYPE_CHECKING, List
from .utils import Serialize
from .lexer import TerminalDef, Token
class ParserConf(Serialize):
    __serialize_fields__ = ('rules', 'start', 'parser_type')
    rules: List['Rule']
    callbacks: ParserCallbacks
    start: List[str]
    parser_type: _ParserArgType

    def __init__(self, rules: List['Rule'], callbacks: ParserCallbacks, start: List[str]):
        assert isinstance(start, list)
        self.rules = rules
        self.callbacks = callbacks
        self.start = start