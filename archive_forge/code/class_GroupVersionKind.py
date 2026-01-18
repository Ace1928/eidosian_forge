from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupVersionKind(_messages.Message):
    """A Kubernetes object's GVK

  Fields:
    group: Kubernetes Group
    kind: Kubernetes Kind
    version: Kubernetes Version
  """
    group = _messages.StringField(1)
    kind = _messages.StringField(2)
    version = _messages.StringField(3)