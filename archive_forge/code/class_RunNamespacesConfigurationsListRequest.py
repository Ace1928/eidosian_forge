from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesConfigurationsListRequest(_messages.Message):
    """A RunNamespacesConfigurationsListRequest object.

  Fields:
    continue_: Optional. Encoded string to continue paging.
    fieldSelector: Not supported by Cloud Run.
    includeUninitialized: Not supported by Cloud Run.
    labelSelector: Allows to filter resources based on a label. Supported
      operations are =, !=, exists, in, and notIn.
    limit: Optional. The maximum number of the records that should be
      returned.
    parent: The namespace from which the configurations should be listed. For
      Cloud Run, replace {namespace_id} with the project ID or number.
    resourceVersion: Not supported by Cloud Run.
    watch: Not supported by Cloud Run.
  """
    continue_ = _messages.StringField(1)
    fieldSelector = _messages.StringField(2)
    includeUninitialized = _messages.BooleanField(3)
    labelSelector = _messages.StringField(4)
    limit = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    parent = _messages.StringField(6, required=True)
    resourceVersion = _messages.StringField(7)
    watch = _messages.BooleanField(8)