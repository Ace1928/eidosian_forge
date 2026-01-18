from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiProxyTypeValueValuesEnum(_messages.Enum):
    """Optional. API Proxy type supported by the environment. The type can be
    set when creating the Environment and cannot be changed.

    Values:
      API_PROXY_TYPE_UNSPECIFIED: API proxy type not specified.
      PROGRAMMABLE: Programmable API Proxies enable you to develop APIs with
        highly flexible behavior using bundled policy configuration and one or
        more programming languages to describe complex sequential and/or
        conditional flows of logic.
      CONFIGURABLE: Configurable API Proxies enable you to develop efficient
        APIs using simple configuration while complex execution control flow
        logic is handled by Apigee. This type only works with the ARCHIVE
        deployment type and cannot be combined with the PROXY deployment type.
    """
    API_PROXY_TYPE_UNSPECIFIED = 0
    PROGRAMMABLE = 1
    CONFIGURABLE = 2