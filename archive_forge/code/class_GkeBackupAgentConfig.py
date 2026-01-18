from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeBackupAgentConfig(_messages.Message):
    """Configuration for the Backup for GKE Agent.

  Fields:
    enabled: Whether the Backup for GKE agent is enabled for this cluster.
  """
    enabled = _messages.BooleanField(1)