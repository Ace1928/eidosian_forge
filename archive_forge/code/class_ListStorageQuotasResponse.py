from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListStorageQuotasResponse(_messages.Message):
    """Response message for the list of Storage provisioning quotas.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    storageQuotas: The provisioning quotas registered in this project.
  """
    nextPageToken = _messages.StringField(1)
    storageQuotas = _messages.MessageField('StorageQuota', 2, repeated=True)