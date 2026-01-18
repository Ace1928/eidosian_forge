from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAccessLevelsDeleteRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAccessLevelsDeleteRequest object.

  Fields:
    name: Required. Resource name for the Access Level. Format:
      `accessPolicies/{policy_id}/accessLevels/{access_level_id}`
  """
    name = _messages.StringField(1, required=True)