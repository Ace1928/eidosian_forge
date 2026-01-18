from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneServiceDirectoryConfigNamespace(_messages.Message):
    """A ManagedZoneServiceDirectoryConfigNamespace object.

  Fields:
    deletionTime: The time that the namespace backing this zone was deleted;
      an empty string if it still exists. This is in RFC3339 text format.
      Output only.
    kind: A string attribute.
    namespaceUrl: The fully qualified URL of the namespace associated with the
      zone. Format must be https://servicedirectory.googleapis.com/v1/projects
      /{project}/locations/{location}/namespaces/{namespace}
  """
    deletionTime = _messages.StringField(1)
    kind = _messages.StringField(2, default='dns#managedZoneServiceDirectoryConfigNamespace')
    namespaceUrl = _messages.StringField(3)