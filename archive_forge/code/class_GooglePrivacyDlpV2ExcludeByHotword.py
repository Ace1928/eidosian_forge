from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ExcludeByHotword(_messages.Message):
    """The rule to exclude findings based on a hotword. For record inspection
  of tables, column names are considered hotwords. An example of this is to
  exclude a finding if it belongs to a BigQuery column that matches a specific
  pattern.

  Fields:
    hotwordRegex: Regular expression pattern defining what qualifies as a
      hotword.
    proximity: Range of characters within which the entire hotword must
      reside. The total length of the window cannot exceed 1000 characters.
      The windowBefore property in proximity should be set to 1 if the hotword
      needs to be included in a column header.
  """
    hotwordRegex = _messages.MessageField('GooglePrivacyDlpV2Regex', 1)
    proximity = _messages.MessageField('GooglePrivacyDlpV2Proximity', 2)