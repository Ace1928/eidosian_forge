from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Pod(_messages.Message):
    """A Kubernetes Pod.

  Fields:
    containers: Pod containers associated with this finding, if any.
    labels: Pod labels. For Kubernetes containers, these are applied to the
      container.
    name: Kubernetes Pod name.
    ns: Kubernetes Pod namespace.
  """
    containers = _messages.MessageField('Container', 1, repeated=True)
    labels = _messages.MessageField('Label', 2, repeated=True)
    name = _messages.StringField(3)
    ns = _messages.StringField(4)