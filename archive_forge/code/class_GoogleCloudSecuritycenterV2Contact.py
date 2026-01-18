from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Contact(_messages.Message):
    """The email address of a contact.

  Fields:
    email: An email address. For example, "`person123@company.com`".
  """
    email = _messages.StringField(1)