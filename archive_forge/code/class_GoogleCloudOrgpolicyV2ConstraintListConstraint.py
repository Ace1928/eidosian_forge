from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2ConstraintListConstraint(_messages.Message):
    """A constraint that allows or disallows a list of string values, which are
  configured by an Organization Policy administrator with a policy.

  Fields:
    supportsIn: Indicates whether values grouped into categories can be used
      in `Policy.allowed_values` and `Policy.denied_values`. For example,
      `"in:Python"` would match any value in the 'Python' group.
    supportsUnder: Indicates whether subtrees of the Resource Manager resource
      hierarchy can be used in `Policy.allowed_values` and
      `Policy.denied_values`. For example, `"under:folders/123"` would match
      any resource under the 'folders/123' folder.
  """
    supportsIn = _messages.BooleanField(1)
    supportsUnder = _messages.BooleanField(2)