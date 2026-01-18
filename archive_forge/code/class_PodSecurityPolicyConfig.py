from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PodSecurityPolicyConfig(_messages.Message):
    """Configuration for the PodSecurityPolicy feature.

  Fields:
    enabled: Enable the PodSecurityPolicy controller for this cluster. If
      enabled, pods must be valid under a PodSecurityPolicy to be created.
  """
    enabled = _messages.BooleanField(1)