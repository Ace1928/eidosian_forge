from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectAgentResource(_messages.Message):
    """ConnectAgentResource represents a Kubernetes resource manifest for
  Connect Agent deployment.

  Fields:
    manifest: YAML manifest of the resource.
    type: Kubernetes type of the resource.
  """
    manifest = _messages.StringField(1)
    type = _messages.MessageField('TypeMeta', 2)