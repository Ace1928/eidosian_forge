from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisksAddResourcePoliciesRequest(_messages.Message):
    """A DisksAddResourcePoliciesRequest object.

  Fields:
    resourcePolicies: Full or relative path to the resource policy to be added
      to this disk. You can only specify one resource policy.
  """
    resourcePolicies = _messages.StringField(1, repeated=True)