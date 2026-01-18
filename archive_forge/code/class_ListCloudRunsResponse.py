from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListCloudRunsResponse(_messages.Message):
    """ListCloudRunsResponse is a list of CloudRun resources. The next page
  token is specified as the "continue" field in ListMeta.

  Fields:
    apiVersion: The API version for this call such as "core/v1".
    items: A CloudRun attribute.
    kind: The kind of this resource, in this case "CloudRunList".
    metadata: Metadata associated with this CloudRun list.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('CloudRun', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)