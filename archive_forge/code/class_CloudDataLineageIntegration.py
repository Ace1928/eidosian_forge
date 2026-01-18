from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudDataLineageIntegration(_messages.Message):
    """Configuration for Cloud Data Lineage integration.

  Fields:
    enabled: Optional. Whether or not Cloud Data Lineage integration is
      enabled.
  """
    enabled = _messages.BooleanField(1)