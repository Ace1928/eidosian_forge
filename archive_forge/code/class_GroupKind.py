from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupKind(_messages.Message):
    """GroupKind includes the group, kind of the K8s resource.

  Fields:
    apiGroup: The api group of the resource.
    kind: The api kind of the resource.
  """
    apiGroup = _messages.StringField(1)
    kind = _messages.StringField(2)