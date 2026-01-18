from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DatabaseResourceCollection(_messages.Message):
    """Match database resources using regex filters. Examples of database
  resources are tables, views, and stored procedures.

  Fields:
    includeRegexes: A collection of regular expressions to match a database
      resource against.
  """
    includeRegexes = _messages.MessageField('GooglePrivacyDlpV2DatabaseResourceRegexes', 1)