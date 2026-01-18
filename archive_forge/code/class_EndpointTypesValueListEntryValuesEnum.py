from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointTypesValueListEntryValuesEnum(_messages.Enum):
    """EndpointTypesValueListEntryValuesEnum enum type.

    Values:
      ENDPOINT_TYPE_MANAGED_PROXY_LB: This is used for regional Application
        Load Balancers (internal and external) and regional proxy Network Load
        Balancers (internal and external) endpoints.
      ENDPOINT_TYPE_SWG: This is used for Secure Web Gateway endpoints.
      ENDPOINT_TYPE_VM: This is the default.
    """
    ENDPOINT_TYPE_MANAGED_PROXY_LB = 0
    ENDPOINT_TYPE_SWG = 1
    ENDPOINT_TYPE_VM = 2