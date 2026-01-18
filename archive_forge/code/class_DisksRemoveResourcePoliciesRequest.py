from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisksRemoveResourcePoliciesRequest(_messages.Message):
    """A DisksRemoveResourcePoliciesRequest object.

  Fields:
    resourcePolicies: Resource policies to be removed from this disk.
  """
    resourcePolicies = _messages.StringField(1, repeated=True)