from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogView(_messages.Message):
    """Describes a view over log entries in a bucket.

  Fields:
    createTime: Output only. The creation timestamp of the view.
    description: Optional. Describes this view.
    filter: Optional. Filter that restricts which log entries in a bucket are
      visible in this view.Filters must be logical conjunctions that use the
      AND operator, and they can use any of the following qualifiers:
      SOURCE(), which specifies a project, folder, organization, or billing
      account of origin. resource.type, which specifies the resource type.
      LOG_ID(), which identifies the log.They can also use the negations of
      these qualifiers with the NOT operator.For
      example:SOURCE("projects/myproject") AND resource.type = "gce_instance"
      AND NOT LOG_ID("stdout")
    name: Output only. The resource name of the view.For example:projects/my-
      project/locations/global/buckets/my-bucket/views/my-view
    schema: Output only. Describes the schema of the logs stored in the bucket
      that are accessible via this view.This field is only populated for views
      in analytics-enabled buckets.
    updateTime: Output only. The last update timestamp of the view.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    filter = _messages.StringField(3)
    name = _messages.StringField(4)
    schema = _messages.MessageField('TableSchema', 5)
    updateTime = _messages.StringField(6)