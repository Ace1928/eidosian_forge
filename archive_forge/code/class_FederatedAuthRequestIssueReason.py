from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class FederatedAuthRequestIssueReason(enum.Enum):
    """
    Represents the failure reason when a federated authentication reason fails.
    Should be updated alongside RequestIdTokenStatus in
    third_party/blink/public/mojom/devtools/inspector_issue.mojom to include
    all cases except for success.
    """
    SHOULD_EMBARGO = 'ShouldEmbargo'
    TOO_MANY_REQUESTS = 'TooManyRequests'
    WELL_KNOWN_HTTP_NOT_FOUND = 'WellKnownHttpNotFound'
    WELL_KNOWN_NO_RESPONSE = 'WellKnownNoResponse'
    WELL_KNOWN_INVALID_RESPONSE = 'WellKnownInvalidResponse'
    WELL_KNOWN_LIST_EMPTY = 'WellKnownListEmpty'
    WELL_KNOWN_INVALID_CONTENT_TYPE = 'WellKnownInvalidContentType'
    CONFIG_NOT_IN_WELL_KNOWN = 'ConfigNotInWellKnown'
    WELL_KNOWN_TOO_BIG = 'WellKnownTooBig'
    CONFIG_HTTP_NOT_FOUND = 'ConfigHttpNotFound'
    CONFIG_NO_RESPONSE = 'ConfigNoResponse'
    CONFIG_INVALID_RESPONSE = 'ConfigInvalidResponse'
    CONFIG_INVALID_CONTENT_TYPE = 'ConfigInvalidContentType'
    CLIENT_METADATA_HTTP_NOT_FOUND = 'ClientMetadataHttpNotFound'
    CLIENT_METADATA_NO_RESPONSE = 'ClientMetadataNoResponse'
    CLIENT_METADATA_INVALID_RESPONSE = 'ClientMetadataInvalidResponse'
    CLIENT_METADATA_INVALID_CONTENT_TYPE = 'ClientMetadataInvalidContentType'
    DISABLED_IN_SETTINGS = 'DisabledInSettings'
    ERROR_FETCHING_SIGNIN = 'ErrorFetchingSignin'
    INVALID_SIGNIN_RESPONSE = 'InvalidSigninResponse'
    ACCOUNTS_HTTP_NOT_FOUND = 'AccountsHttpNotFound'
    ACCOUNTS_NO_RESPONSE = 'AccountsNoResponse'
    ACCOUNTS_INVALID_RESPONSE = 'AccountsInvalidResponse'
    ACCOUNTS_LIST_EMPTY = 'AccountsListEmpty'
    ACCOUNTS_INVALID_CONTENT_TYPE = 'AccountsInvalidContentType'
    ID_TOKEN_HTTP_NOT_FOUND = 'IdTokenHttpNotFound'
    ID_TOKEN_NO_RESPONSE = 'IdTokenNoResponse'
    ID_TOKEN_INVALID_RESPONSE = 'IdTokenInvalidResponse'
    ID_TOKEN_IDP_ERROR_RESPONSE = 'IdTokenIdpErrorResponse'
    ID_TOKEN_CROSS_SITE_IDP_ERROR_RESPONSE = 'IdTokenCrossSiteIdpErrorResponse'
    ID_TOKEN_INVALID_REQUEST = 'IdTokenInvalidRequest'
    ID_TOKEN_INVALID_CONTENT_TYPE = 'IdTokenInvalidContentType'
    ERROR_ID_TOKEN = 'ErrorIdToken'
    CANCELED = 'Canceled'
    RP_PAGE_NOT_VISIBLE = 'RpPageNotVisible'
    SILENT_MEDIATION_FAILURE = 'SilentMediationFailure'
    THIRD_PARTY_COOKIES_BLOCKED = 'ThirdPartyCookiesBlocked'
    NOT_SIGNED_IN_WITH_IDP = 'NotSignedInWithIdp'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)