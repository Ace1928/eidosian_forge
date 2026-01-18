from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResponseTypeValueValuesEnum(_messages.Enum):
    """Required. The Response Type to request for in the OIDC Authorization
    Request for web sign-in. The `CODE` Response Type is recommended to avoid
    the Implicit Flow, for security reasons.

    Values:
      RESPONSE_TYPE_UNSPECIFIED: No Response Type specified.
      CODE: The `response_type=code` selection uses the Authorization Code
        Flow for web sign-in. Requires a configured client secret.
      ID_TOKEN: The `response_type=id_token` selection uses the Implicit Flow
        for web sign-in.
    """
    RESPONSE_TYPE_UNSPECIFIED = 0
    CODE = 1
    ID_TOKEN = 2