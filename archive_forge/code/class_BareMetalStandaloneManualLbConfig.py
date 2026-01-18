from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneManualLbConfig(_messages.Message):
    """Represents configuration parameters for a manual load balancer.

  Fields:
    enabled: Whether manual load balancing is enabled.
  """
    enabled = _messages.BooleanField(1)