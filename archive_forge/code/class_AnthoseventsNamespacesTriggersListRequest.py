from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesTriggersListRequest(_messages.Message):
    """A AnthoseventsNamespacesTriggersListRequest object.

  Fields:
    continue_: Encoded string to continue paging.
    fieldSelector: Allows to filter resources based on a specific value for a
      field name. Send this in a query string format. i.e.
      'metadata.name%3Dlorem'. Not currently used by Events for Cloud Run.
    includeUninitialized: If true, partially initialized resources are
      included in the response. Not currently used by Events for Cloud Run.
    labelSelector: Allows to filter resources based on a label. Supported
      operations are =, !=, exists, in, and notIn.
    pageSize: The maximum number of records that should be returned.
    parent: The namespace name.
    resourceVersion: The baseline resource version from which the list or
      watch operation should start. Not currently used by Events for Cloud
      Run.
    watch: Flag that indicates that the client expects to watch this resource
      as well. Not currently used by Events for Cloud Run.
  """
    continue_ = _messages.StringField(1)
    fieldSelector = _messages.StringField(2)
    includeUninitialized = _messages.BooleanField(3)
    labelSelector = _messages.StringField(4)
    pageSize = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    parent = _messages.StringField(6, required=True)
    resourceVersion = _messages.StringField(7)
    watch = _messages.BooleanField(8)