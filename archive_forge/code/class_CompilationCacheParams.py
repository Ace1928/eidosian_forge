from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class CompilationCacheParams:
    """
    Per-script compilation cache parameters for ``Page.produceCompilationCache``
    """
    url: str
    eager: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.eager is not None:
            json['eager'] = self.eager
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), eager=bool(json['eager']) if 'eager' in json else None)