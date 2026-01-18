from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PublishToStackdriver(_messages.Message):
    """Enable Stackdriver metric dlp.googleapis.com/finding_count. This will
  publish a metric to stack driver on each infotype requested and how many
  findings were found for it. CustomDetectors will be bucketed as 'Custom'
  under the Stackdriver label 'info_type'.
  """