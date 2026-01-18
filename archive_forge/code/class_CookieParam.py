from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class CookieParam:
    """
    Cookie parameter object
    """
    name: str
    value: str
    url: typing.Optional[str] = None
    domain: typing.Optional[str] = None
    path: typing.Optional[str] = None
    secure: typing.Optional[bool] = None
    http_only: typing.Optional[bool] = None
    same_site: typing.Optional[CookieSameSite] = None
    expires: typing.Optional[TimeSinceEpoch] = None
    priority: typing.Optional[CookiePriority] = None
    same_party: typing.Optional[bool] = None
    source_scheme: typing.Optional[CookieSourceScheme] = None
    source_port: typing.Optional[int] = None
    partition_key: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        if self.url is not None:
            json['url'] = self.url
        if self.domain is not None:
            json['domain'] = self.domain
        if self.path is not None:
            json['path'] = self.path
        if self.secure is not None:
            json['secure'] = self.secure
        if self.http_only is not None:
            json['httpOnly'] = self.http_only
        if self.same_site is not None:
            json['sameSite'] = self.same_site.to_json()
        if self.expires is not None:
            json['expires'] = self.expires.to_json()
        if self.priority is not None:
            json['priority'] = self.priority.to_json()
        if self.same_party is not None:
            json['sameParty'] = self.same_party
        if self.source_scheme is not None:
            json['sourceScheme'] = self.source_scheme.to_json()
        if self.source_port is not None:
            json['sourcePort'] = self.source_port
        if self.partition_key is not None:
            json['partitionKey'] = self.partition_key
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), value=str(json['value']), url=str(json['url']) if 'url' in json else None, domain=str(json['domain']) if 'domain' in json else None, path=str(json['path']) if 'path' in json else None, secure=bool(json['secure']) if 'secure' in json else None, http_only=bool(json['httpOnly']) if 'httpOnly' in json else None, same_site=CookieSameSite.from_json(json['sameSite']) if 'sameSite' in json else None, expires=TimeSinceEpoch.from_json(json['expires']) if 'expires' in json else None, priority=CookiePriority.from_json(json['priority']) if 'priority' in json else None, same_party=bool(json['sameParty']) if 'sameParty' in json else None, source_scheme=CookieSourceScheme.from_json(json['sourceScheme']) if 'sourceScheme' in json else None, source_port=int(json['sourcePort']) if 'sourcePort' in json else None, partition_key=str(json['partitionKey']) if 'partitionKey' in json else None)