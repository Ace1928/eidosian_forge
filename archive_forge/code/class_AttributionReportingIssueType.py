from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class AttributionReportingIssueType(enum.Enum):
    PERMISSION_POLICY_DISABLED = 'PermissionPolicyDisabled'
    UNTRUSTWORTHY_REPORTING_ORIGIN = 'UntrustworthyReportingOrigin'
    INSECURE_CONTEXT = 'InsecureContext'
    INVALID_HEADER = 'InvalidHeader'
    INVALID_REGISTER_TRIGGER_HEADER = 'InvalidRegisterTriggerHeader'
    SOURCE_AND_TRIGGER_HEADERS = 'SourceAndTriggerHeaders'
    SOURCE_IGNORED = 'SourceIgnored'
    TRIGGER_IGNORED = 'TriggerIgnored'
    OS_SOURCE_IGNORED = 'OsSourceIgnored'
    OS_TRIGGER_IGNORED = 'OsTriggerIgnored'
    INVALID_REGISTER_OS_SOURCE_HEADER = 'InvalidRegisterOsSourceHeader'
    INVALID_REGISTER_OS_TRIGGER_HEADER = 'InvalidRegisterOsTriggerHeader'
    WEB_AND_OS_HEADERS = 'WebAndOsHeaders'
    NO_WEB_OR_OS_SUPPORT = 'NoWebOrOsSupport'
    NAVIGATION_REGISTRATION_WITHOUT_TRANSIENT_USER_ACTIVATION = 'NavigationRegistrationWithoutTransientUserActivation'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)