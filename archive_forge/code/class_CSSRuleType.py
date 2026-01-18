from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
class CSSRuleType(enum.Enum):
    """
    Enum indicating the type of a CSS rule, used to represent the order of a style rule's ancestors.
    This list only contains rule types that are collected during the ancestor rule collection.
    """
    MEDIA_RULE = 'MediaRule'
    SUPPORTS_RULE = 'SupportsRule'
    CONTAINER_RULE = 'ContainerRule'
    LAYER_RULE = 'LayerRule'
    SCOPE_RULE = 'ScopeRule'
    STYLE_RULE = 'StyleRule'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)