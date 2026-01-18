from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultClusterConfig(_messages.Message):
    """DefaultClusterConfig describes the default cluster configurations to be
  applied to all clusters born-in-fleet.

  Fields:
    binaryAuthorizationConfig: Optional. Enable/Disable binary authorization
      features for the cluster.
    compliancePostureConfig: A CompliancePostureConfig attribute.
    securityPostureConfig: Enable/Disable Security Posture features for the
      cluster.
  """
    binaryAuthorizationConfig = _messages.MessageField('BinaryAuthorizationConfig', 1)
    compliancePostureConfig = _messages.MessageField('CompliancePostureConfig', 2)
    securityPostureConfig = _messages.MessageField('SecurityPostureConfig', 3)