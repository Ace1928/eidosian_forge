from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1ValueMatcher(_messages.Message):
    """Contains various matching options for values for a GCP resource field.

  Fields:
    matchesPattern: To be used for full regex matching. The regular expression
      is using the Google RE2 syntax
      (https://github.com/google/re2/wiki/Syntax), so to be used with
      RE2::FullMatch
  """
    matchesPattern = _messages.StringField(1)