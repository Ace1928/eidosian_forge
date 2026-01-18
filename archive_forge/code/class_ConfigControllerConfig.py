from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigControllerConfig(_messages.Message):
    """Configuration options for the Config Controller bundle.

  Fields:
    enabled: Whether the Config Controller bundle is enabled on the
      KrmApiHost.
  """
    enabled = _messages.BooleanField(1)