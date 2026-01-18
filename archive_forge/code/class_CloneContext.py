from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloneContext(_messages.Message):
    """Database instance clone context.

  Fields:
    allocatedIpRange: The name of the allocated ip range for the private ip
      Cloud SQL instance. For example: "google-managed-services-default". If
      set, the cloned instance ip will be created in the allocated range. The
      range name must comply with [RFC
      1035](https://tools.ietf.org/html/rfc1035). Specifically, the name must
      be 1-63 characters long and match the regular expression
      [a-z]([-a-z0-9]*[a-z0-9])?. Reserved for future use.
    binLogCoordinates: Binary log coordinates, if specified, identify the
      position up to which the source instance is cloned. If not specified,
      the source instance is cloned up to the most recent binary log
      coordinates.
    databaseNames: (SQL Server only) Clone only the specified databases from
      the source instance. Clone all databases if empty.
    destinationInstanceName: Name of the Cloud SQL instance to be created as a
      clone.
    kind: This is always `sql#cloneContext`.
    pitrTimestampMs: Reserved for future use.
    pointInTime: Timestamp, if specified, identifies the time to which the
      source instance is cloned.
    preferredZone: Optional. (Point-in-time recovery for PostgreSQL only)
      Clone to an instance in the specified zone. If no zone is specified,
      clone to the same zone as the source instance.
  """
    allocatedIpRange = _messages.StringField(1)
    binLogCoordinates = _messages.MessageField('BinLogCoordinates', 2)
    databaseNames = _messages.StringField(3, repeated=True)
    destinationInstanceName = _messages.StringField(4)
    kind = _messages.StringField(5)
    pitrTimestampMs = _messages.IntegerField(6)
    pointInTime = _messages.StringField(7)
    preferredZone = _messages.StringField(8)