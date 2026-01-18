from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetOtherCloudConnectionsDeleteRequest(_messages.Message):
    """A CloudassetOtherCloudConnectionsDeleteRequest object.

  Fields:
    name: Required. The name of the other-cloud connection to delete. Format:
      organizations/{organization_number}/otherCloudConnections/{other_cloud_c
      onnection_id}
  """
    name = _messages.StringField(1, required=True)