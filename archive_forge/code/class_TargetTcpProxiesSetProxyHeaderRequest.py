from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetTcpProxiesSetProxyHeaderRequest(_messages.Message):
    """A TargetTcpProxiesSetProxyHeaderRequest object.

  Enums:
    ProxyHeaderValueValuesEnum: The new type of proxy header to append before
      sending data to the backend. NONE or PROXY_V1 are allowed.

  Fields:
    proxyHeader: The new type of proxy header to append before sending data to
      the backend. NONE or PROXY_V1 are allowed.
  """

    class ProxyHeaderValueValuesEnum(_messages.Enum):
        """The new type of proxy header to append before sending data to the
    backend. NONE or PROXY_V1 are allowed.

    Values:
      NONE: <no description>
      PROXY_V1: <no description>
    """
        NONE = 0
        PROXY_V1 = 1
    proxyHeader = _messages.EnumField('ProxyHeaderValueValuesEnum', 1)