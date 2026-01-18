from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class FederatedAuthUserInfoRequestIssueReason(enum.Enum):
    """
    Represents the failure reason when a getUserInfo() call fails.
    Should be updated alongside FederatedAuthUserInfoRequestResult in
    third_party/blink/public/mojom/devtools/inspector_issue.mojom.
    """
    NOT_SAME_ORIGIN = 'NotSameOrigin'
    NOT_IFRAME = 'NotIframe'
    NOT_POTENTIALLY_TRUSTWORTHY = 'NotPotentiallyTrustworthy'
    NO_API_PERMISSION = 'NoApiPermission'
    NOT_SIGNED_IN_WITH_IDP = 'NotSignedInWithIdp'
    NO_ACCOUNT_SHARING_PERMISSION = 'NoAccountSharingPermission'
    INVALID_CONFIG_OR_WELL_KNOWN = 'InvalidConfigOrWellKnown'
    INVALID_ACCOUNTS_RESPONSE = 'InvalidAccountsResponse'
    NO_RETURNING_USER_FROM_FETCHED_ACCOUNTS = 'NoReturningUserFromFetchedAccounts'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)