from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryRegexes(_messages.Message):
    """A collection of regular expressions to determine what tables to match
  against.

  Fields:
    patterns: A single BigQuery regular expression pattern to match against
      one or more tables, datasets, or projects that contain BigQuery tables.
  """
    patterns = _messages.MessageField('GooglePrivacyDlpV2BigQueryRegex', 1, repeated=True)