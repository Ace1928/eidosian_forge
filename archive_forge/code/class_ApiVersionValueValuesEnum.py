from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiVersionValueValuesEnum(_messages.Enum):
    """Optional. API version displayed in Dialogflow console. If not
    specified, V2 API is assumed. Clients are free to query different service
    endpoints for different API versions. However, bots connectors and webhook
    calls will follow the specified API version.

    Values:
      API_VERSION_UNSPECIFIED: Not specified.
      API_VERSION_V1: Legacy V1 API.
      API_VERSION_V2: V2 API.
      API_VERSION_V2_BETA_1: V2beta1 API.
    """
    API_VERSION_UNSPECIFIED = 0
    API_VERSION_V1 = 1
    API_VERSION_V2 = 2
    API_VERSION_V2_BETA_1 = 3