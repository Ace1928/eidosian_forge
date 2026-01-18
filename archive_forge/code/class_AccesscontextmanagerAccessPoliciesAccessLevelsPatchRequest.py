from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAccessLevelsPatchRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAccessLevelsPatchRequest object.

  Fields:
    accessLevel: A AccessLevel resource to be passed as the request body.
    name: Required. Resource name for the `AccessLevel`. Format:
      `accessPolicies/{access_policy}/accessLevels/{access_level}`. The
      `access_level` component must begin with a letter, followed by
      alphanumeric characters or `_`. Its maximum length is 50 characters.
      After you create an `AccessLevel`, you cannot change its `name`.
    updateMask: Required. Mask to control which fields get updated. Must be
      non-empty.
  """
    accessLevel = _messages.MessageField('AccessLevel', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)