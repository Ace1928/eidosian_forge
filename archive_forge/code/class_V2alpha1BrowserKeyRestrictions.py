from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1BrowserKeyRestrictions(_messages.Message):
    """Key restrictions that are specific to browser keys.

  Fields:
    allowedReferrers: A list of regular expressions for the referrer URLs that
      are allowed when making an API call with this key.
  """
    allowedReferrers = _messages.StringField(1, repeated=True)