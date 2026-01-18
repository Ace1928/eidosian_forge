from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class PreloadingAttemptSource:
    """
    Lists sources for a preloading attempt, specifically the ids of rule sets
    that had a speculation rule that triggered the attempt, and the
    BackendNodeIds of <a href> or <area href> elements that triggered the
    attempt (in the case of attempts triggered by a document rule). It is
    possible for mulitple rule sets and links to trigger a single attempt.
    """
    key: PreloadingAttemptKey
    rule_set_ids: typing.List[RuleSetId]
    node_ids: typing.List[dom.BackendNodeId]

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['ruleSetIds'] = [i.to_json() for i in self.rule_set_ids]
        json['nodeIds'] = [i.to_json() for i in self.node_ids]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(key=PreloadingAttemptKey.from_json(json['key']), rule_set_ids=[RuleSetId.from_json(i) for i in json['ruleSetIds']], node_ids=[dom.BackendNodeId.from_json(i) for i in json['nodeIds']])