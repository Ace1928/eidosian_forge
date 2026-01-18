from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaGroup(_messages.Message):
    """Message to capture group information

  Fields:
    email: The group email id
    id: Google group id
  """
    email = _messages.StringField(1)
    id = _messages.StringField(2)