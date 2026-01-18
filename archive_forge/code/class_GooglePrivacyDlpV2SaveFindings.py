from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SaveFindings(_messages.Message):
    """If set, the detailed findings will be persisted to the specified
  OutputStorageConfig. Only a single instance of this action can be specified.
  Compatible with: Inspect, Risk

  Fields:
    outputConfig: Location to store findings outside of DLP.
  """
    outputConfig = _messages.MessageField('GooglePrivacyDlpV2OutputStorageConfig', 1)