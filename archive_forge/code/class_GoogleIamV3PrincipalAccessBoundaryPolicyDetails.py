from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3PrincipalAccessBoundaryPolicyDetails(_messages.Message):
    """Principal access boundary policy details

  Fields:
    enforcementVersion: Optional. The version number that indicates which GCP
      services are included in the enforcement (e.g. "latest", "1", ...). If
      empty, the PAB policy version will be set to the current latest version,
      and this version won't get updated when new versions are released.
    rules: Required. A list of principal access boundary policy rules.
  """
    enforcementVersion = _messages.StringField(1)
    rules = _messages.MessageField('GoogleIamV3PrincipalAccessBoundaryPolicyRule', 2, repeated=True)