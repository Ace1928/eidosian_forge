from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListProvisioningQuotasResponse(_messages.Message):
    """Response message for the list of provisioning quotas.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    provisioningQuotas: The provisioning quotas registered in this project.
  """
    nextPageToken = _messages.StringField(1)
    provisioningQuotas = _messages.MessageField('ProvisioningQuota', 2, repeated=True)