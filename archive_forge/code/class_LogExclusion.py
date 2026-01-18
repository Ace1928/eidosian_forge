from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogExclusion(_messages.Message):
    """Specifies a set of log entries that are filtered out by a sink. If your
  Google Cloud resource receives a large volume of log entries, you can use
  exclusions to reduce your chargeable logs. Note that exclusions on
  organization-level and folder-level sinks don't apply to child resources.
  Note also that you cannot modify the _Required sink or exclude logs from it.

  Fields:
    createTime: Output only. The creation timestamp of the exclusion.This
      field may not be present for older exclusions.
    description: Optional. A description of this exclusion.
    disabled: Optional. If set to True, then this exclusion is disabled and it
      does not exclude any log entries. You can update an exclusion to change
      the value of this field.
    filter: Required. An advanced logs filter
      (https://cloud.google.com/logging/docs/view/advanced-queries) that
      matches the log entries to be excluded. By using the sample function
      (https://cloud.google.com/logging/docs/view/advanced-queries#sample),
      you can exclude less than 100% of the matching log entries.For example,
      the following query matches 99% of low-severity log entries from Google
      Cloud Storage buckets:resource.type=gcs_bucket severity<ERROR
      sample(insertId, 0.99)
    name: Output only. A client-assigned identifier, such as "load-balancer-
      exclusion". Identifiers are limited to 100 characters and can include
      only letters, digits, underscores, hyphens, and periods. First character
      has to be alphanumeric.
    updateTime: Output only. The last update timestamp of the exclusion.This
      field may not be present for older exclusions.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    disabled = _messages.BooleanField(3)
    filter = _messages.StringField(4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)