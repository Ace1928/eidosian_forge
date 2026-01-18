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
class CookieExemptionReason(enum.Enum):
    """
    Types of reasons why a cookie should have been blocked by 3PCD but is exempted for the request.
    """
    NONE = 'None'
    USER_SETTING = 'UserSetting'
    TPCD_METADATA = 'TPCDMetadata'
    TPCD_DEPRECATION_TRIAL = 'TPCDDeprecationTrial'
    TPCD_HEURISTICS = 'TPCDHeuristics'
    ENTERPRISE_POLICY = 'EnterprisePolicy'
    STORAGE_ACCESS = 'StorageAccess'
    TOP_LEVEL_STORAGE_ACCESS = 'TopLevelStorageAccess'
    CORS_OPT_IN = 'CorsOptIn'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)