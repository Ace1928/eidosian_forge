from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagsValueListEntry(_messages.Message):
    """A global tag managed by Resource Manager.
    https://cloud.google.com/iam/docs/tags-access-control#definitions

    Fields:
      tagKey: Required. The namespaced friendly name of the tag key, e.g.
        "12345/environment" where 12345 is org id.
      tagValue: Required. The friendly short name of the tag value, e.g.
        "production".
    """
    tagKey = _messages.StringField(1)
    tagValue = _messages.StringField(2)