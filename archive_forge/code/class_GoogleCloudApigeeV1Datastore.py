from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Datastore(_messages.Message):
    """The data store defines the connection to export data repository (Cloud
  Storage, BigQuery), including the credentials used to access the data
  repository.

  Fields:
    createTime: Output only. Datastore create time, in milliseconds since the
      epoch of 1970-01-01T00:00:00Z
    datastoreConfig: Datastore Configurations.
    displayName: Required. Display name in UI
    lastUpdateTime: Output only. Datastore last update time, in milliseconds
      since the epoch of 1970-01-01T00:00:00Z
    org: Output only. Organization that the datastore belongs to
    self: Output only. Resource link of Datastore. Example:
      `/organizations/{org}/analytics/datastores/{uuid}`
    targetType: Destination storage type. Supported types `gcs` or `bigquery`.
  """
    createTime = _messages.IntegerField(1)
    datastoreConfig = _messages.MessageField('GoogleCloudApigeeV1DatastoreConfig', 2)
    displayName = _messages.StringField(3)
    lastUpdateTime = _messages.IntegerField(4)
    org = _messages.StringField(5)
    self = _messages.StringField(6)
    targetType = _messages.StringField(7)