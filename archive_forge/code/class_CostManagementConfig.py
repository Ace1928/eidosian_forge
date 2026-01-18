from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CostManagementConfig(_messages.Message):
    """Configuration for fine-grained cost management feature.

  Fields:
    enabled: Whether the feature is enabled or not.
  """
    enabled = _messages.BooleanField(1)