from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadALTSConfig(_messages.Message):
    """Configuration for direct-path (via ALTS) with workload identity.

  Fields:
    enableAlts: enable_alts controls whether the alts handshaker should be
      enabled or not for direct-path. Requires Workload Identity
      (workload_pool must be non-empty).
  """
    enableAlts = _messages.BooleanField(1)