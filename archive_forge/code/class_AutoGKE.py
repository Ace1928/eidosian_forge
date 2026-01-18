from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoGKE(_messages.Message):
    """AutoGKE is the configuration for AutoGKE settings on the cluster.
  Replaced by Autopilot.

  Fields:
    enabled: Enable AutoGKE
  """
    enabled = _messages.BooleanField(1)