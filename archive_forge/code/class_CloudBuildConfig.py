from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudBuildConfig(_messages.Message):
    """Configuration options for the Cloud Build addon.

  Fields:
    enabled: Whether the Cloud Build addon is enabled for this cluster.
  """
    enabled = _messages.BooleanField(1)