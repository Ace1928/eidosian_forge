from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
@dataclass
class ApplicationCache:
    """
    Detailed application cache information.
    """
    manifest_url: str
    size: float
    creation_time: float
    update_time: float
    resources: typing.List[ApplicationCacheResource]

    def to_json(self):
        json = dict()
        json['manifestURL'] = self.manifest_url
        json['size'] = self.size
        json['creationTime'] = self.creation_time
        json['updateTime'] = self.update_time
        json['resources'] = [i.to_json() for i in self.resources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(manifest_url=str(json['manifestURL']), size=float(json['size']), creation_time=float(json['creationTime']), update_time=float(json['updateTime']), resources=[ApplicationCacheResource.from_json(i) for i in json['resources']])