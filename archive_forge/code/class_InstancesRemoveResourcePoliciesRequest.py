from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesRemoveResourcePoliciesRequest(_messages.Message):
    """A InstancesRemoveResourcePoliciesRequest object.

  Fields:
    resourcePolicies: Resource policies to be removed from this instance.
  """
    resourcePolicies = _messages.StringField(1, repeated=True)