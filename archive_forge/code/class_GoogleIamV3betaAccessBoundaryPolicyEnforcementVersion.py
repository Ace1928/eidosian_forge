from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaAccessBoundaryPolicyEnforcementVersion(_messages.Message):
    """The different versions supported by principal authorization policy.

  Fields:
    latestVersion: Optional. The latest officially released version number.
      This will automatically increase in scope as more services are included
      in enforcemment.
    staticVersion: Optional. A specific version number. This will need to be
      manually updated to newer versions as they become available in order to
      keep maximum protection.
  """
    latestVersion = _messages.MessageField('GoogleIamV3betaAccessBoundaryPolicyEnforcementVersionLatestVersion', 1)
    staticVersion = _messages.MessageField('GoogleIamV3betaAccessBoundaryPolicyEnforcementVersionStaticVersion', 2)