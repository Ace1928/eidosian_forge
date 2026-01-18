from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespacedName(_messages.Message):
    """A reference to a namespaced resource in Kubernetes.

  Fields:
    name: Optional. The name of the Kubernetes resource.
    namespace: Optional. The Namespace of the Kubernetes resource.
  """
    name = _messages.StringField(1)
    namespace = _messages.StringField(2)