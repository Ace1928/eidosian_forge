from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ControlPlane(_messages.Message):
    """Configuration of the cluster control plane.

  Fields:
    local: Local control plane configuration. Warning: Local control plane
      clusters must be created in their own project. Local control plane
      clusters cannot coexist in the same project with any other type of
      clusters, including non-GDCE clusters. Mixing local control plane GDCE
      clusters with any other type of clusters in the same project can result
      in data loss.
    remote: Remote control plane configuration.
  """
    local = _messages.MessageField('Local', 1)
    remote = _messages.MessageField('Remote', 2)