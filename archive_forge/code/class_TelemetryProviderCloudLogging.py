from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelemetryProviderCloudLogging(_messages.Message):
    """Google Cloud Logging provider configuration for access logging. Default
  (empty) configuration sends access logging to Google Cloud Logging.
  """