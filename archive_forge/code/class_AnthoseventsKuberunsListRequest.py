from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsKuberunsListRequest(_messages.Message):
    """A AnthoseventsKuberunsListRequest object.

  Fields:
    continue_: Optional encoded string to continue paging.
    fieldSelector: Allows to filter resources based on a specific value for a
      field name. Send this in a query string format. i.e.
      'metadata.name%3Dlorem'.
    labelSelector: Allows to filter resources based on a label. Supported
      operations are =, !=, exists, in, and notIn.
    limit: The maximum number of records that should be returned.
    parent: The namespace from which the KubeRun resources should be listed.
    resourceVersion: The baseline resource version from which the list or
      watch operation should start.
    watch: Flag that indicates that the client expects to watch this resource
      as well.
  """
    continue_ = _messages.StringField(1)
    fieldSelector = _messages.StringField(2)
    labelSelector = _messages.StringField(3)
    limit = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    parent = _messages.StringField(5)
    resourceVersion = _messages.StringField(6)
    watch = _messages.BooleanField(7)