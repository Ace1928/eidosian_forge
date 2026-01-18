from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSqlConnectionProfile(_messages.Message):
    """Specifies required connection parameters, and, optionally, the
  parameters required to create a Cloud SQL destination database instance.

  Fields:
    cloudSqlId: Output only. The Cloud SQL instance ID that this connection
      profile is associated with.
    privateIp: Output only. The Cloud SQL database instance's private IP.
    publicIp: Output only. The Cloud SQL database instance's public IP.
    settings: Immutable. Metadata used to create the destination Cloud SQL
      database.
  """
    cloudSqlId = _messages.StringField(1)
    privateIp = _messages.StringField(2)
    publicIp = _messages.StringField(3)
    settings = _messages.MessageField('CloudSqlSettings', 4)