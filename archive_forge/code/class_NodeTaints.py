from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeTaints(_messages.Message):
    """Collection of Kubernetes [node
  taints](https://kubernetes.io/docs/concepts/configuration/taint-and-
  toleration).

  Fields:
    taints: List of node taints.
  """
    taints = _messages.MessageField('NodeTaint', 1, repeated=True)