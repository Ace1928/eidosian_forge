from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class AXValueType(enum.Enum):
    """
    Enum of possible property types.
    """
    BOOLEAN = 'boolean'
    TRISTATE = 'tristate'
    BOOLEAN_OR_UNDEFINED = 'booleanOrUndefined'
    IDREF = 'idref'
    IDREF_LIST = 'idrefList'
    INTEGER = 'integer'
    NODE = 'node'
    NODE_LIST = 'nodeList'
    NUMBER = 'number'
    STRING = 'string'
    COMPUTED_STRING = 'computedString'
    TOKEN = 'token'
    TOKEN_LIST = 'tokenList'
    DOM_RELATION = 'domRelation'
    ROLE = 'role'
    INTERNAL_ROLE = 'internalRole'
    VALUE_UNDEFINED = 'valueUndefined'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)