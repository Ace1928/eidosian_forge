from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlMapReference(_messages.Message):
    """A UrlMapReference object.

  Fields:
    urlMap: A string attribute.
  """
    urlMap = _messages.StringField(1)