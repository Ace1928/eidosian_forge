from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProxyHeaderValueValuesEnum(_messages.Enum):
    """Specifies the type of proxy header to append before sending data to
    the backend, either NONE or PROXY_V1. The default is NONE.

    Values:
      NONE: <no description>
      PROXY_V1: <no description>
    """
    NONE = 0
    PROXY_V1 = 1