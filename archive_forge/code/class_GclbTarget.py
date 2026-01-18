from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GclbTarget(_messages.Message):
    """Describes a Target Proxy which uses this Certificate Map.

  Fields:
    ipConfigs: IP configurations for this Target Proxy where the Certificate
      Map is serving.
    targetHttpsProxy: Output only. This field returns the resource name in the
      following format:
      `//compute.googleapis.com/projects/*/global/targetHttpsProxies/*`.
    targetSslProxy: Output only. This field returns the resource name in the
      following format:
      `//compute.googleapis.com/projects/*/global/targetSslProxies/*`.
  """
    ipConfigs = _messages.MessageField('IpConfig', 1, repeated=True)
    targetHttpsProxy = _messages.StringField(2)
    targetSslProxy = _messages.StringField(3)