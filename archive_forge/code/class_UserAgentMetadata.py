from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class UserAgentMetadata:
    """
    Used to specify User Agent Cient Hints to emulate. See https://wicg.github.io/ua-client-hints
    Missing optional values will be filled in by the target with what it would normally use.
    """
    platform: str
    platform_version: str
    architecture: str
    model: str
    mobile: bool
    brands: typing.Optional[typing.List[UserAgentBrandVersion]] = None
    full_version_list: typing.Optional[typing.List[UserAgentBrandVersion]] = None
    full_version: typing.Optional[str] = None
    bitness: typing.Optional[str] = None
    wow64: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['platform'] = self.platform
        json['platformVersion'] = self.platform_version
        json['architecture'] = self.architecture
        json['model'] = self.model
        json['mobile'] = self.mobile
        if self.brands is not None:
            json['brands'] = [i.to_json() for i in self.brands]
        if self.full_version_list is not None:
            json['fullVersionList'] = [i.to_json() for i in self.full_version_list]
        if self.full_version is not None:
            json['fullVersion'] = self.full_version
        if self.bitness is not None:
            json['bitness'] = self.bitness
        if self.wow64 is not None:
            json['wow64'] = self.wow64
        return json

    @classmethod
    def from_json(cls, json):
        return cls(platform=str(json['platform']), platform_version=str(json['platformVersion']), architecture=str(json['architecture']), model=str(json['model']), mobile=bool(json['mobile']), brands=[UserAgentBrandVersion.from_json(i) for i in json['brands']] if 'brands' in json else None, full_version_list=[UserAgentBrandVersion.from_json(i) for i in json['fullVersionList']] if 'fullVersionList' in json else None, full_version=str(json['fullVersion']) if 'fullVersion' in json else None, bitness=str(json['bitness']) if 'bitness' in json else None, wow64=bool(json['wow64']) if 'wow64' in json else None)