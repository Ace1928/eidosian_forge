from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstanceOSPoliciesCompliancesResponse(_messages.Message):
    """A response message for listing OS policies compliance data for all
  Compute Engine VMs in the given location.

  Fields:
    instanceOsPoliciesCompliances: List of instance OS policies compliance
      objects.
    nextPageToken: The pagination token to retrieve the next page of instance
      OS policies compliance objects.
  """
    instanceOsPoliciesCompliances = _messages.MessageField('InstanceOSPoliciesCompliance', 1, repeated=True)
    nextPageToken = _messages.StringField(2)