from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class PseudoElementMatches:
    """
    CSS rule collection for a single pseudo style.
    """
    pseudo_type: dom.PseudoType
    matches: typing.List[RuleMatch]
    pseudo_identifier: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['pseudoType'] = self.pseudo_type.to_json()
        json['matches'] = [i.to_json() for i in self.matches]
        if self.pseudo_identifier is not None:
            json['pseudoIdentifier'] = self.pseudo_identifier
        return json

    @classmethod
    def from_json(cls, json):
        return cls(pseudo_type=dom.PseudoType.from_json(json['pseudoType']), matches=[RuleMatch.from_json(i) for i in json['matches']], pseudo_identifier=str(json['pseudoIdentifier']) if 'pseudoIdentifier' in json else None)