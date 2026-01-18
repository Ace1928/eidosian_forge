from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1PersonalDetails(_messages.Message):
    """Entry metadata relevant only to the user and private to them.

  Fields:
    starTime: Set if the entry is starred; unset otherwise.
    starred: True if the entry is starred by the user; false otherwise.
  """
    starTime = _messages.StringField(1)
    starred = _messages.BooleanField(2)