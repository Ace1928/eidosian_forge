from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoredResourceDescriptor(_messages.Message):
    """An object that describes the schema of a MonitoredResource object using
  a type name and a set of labels.  For example, the monitored resource
  descriptor for Google Compute Engine VM instances has a type of
  `"gce_instance"` and specifies the use of the labels `"instance_id"` and
  `"zone"` to identify particular VM instances.  Different APIs can support
  different monitored resource types. APIs generally provide a `list` method
  that returns the monitored resource descriptors used by the API.

  Fields:
    description: Optional. A detailed description of the monitored resource
      type that might be used in documentation.
    displayName: Optional. A concise name for the monitored resource type that
      might be displayed in user interfaces. For example, `"Google Cloud SQL
      Database"`.
    labels: Required. A set of labels used to describe instances of this
      monitored resource type. For example, an individual Google Cloud SQL
      database is identified by values for the labels `"database_id"` and
      `"zone"`.
    name: Optional. The resource name of the monitored resource descriptor:
      `"projects/{project_id}/monitoredResourceDescriptors/{type}"` where
      {type} is the value of the `type` field in this object and {project_id}
      is a project ID that provides API-specific context for accessing the
      type.  APIs that do not use project information can use the resource
      name format `"monitoredResourceDescriptors/{type}"`.
    type: Required. The monitored resource type. For example, the type
      `"cloudsql_database"` represents databases in Google Cloud SQL. The
      maximum length of this value is 256 characters.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelDescriptor', 3, repeated=True)
    name = _messages.StringField(4)
    type = _messages.StringField(5)