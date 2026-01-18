from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersProjectsZonesSignSshPublicKeyRequest(_messages.Message):
    """A OsloginUsersProjectsZonesSignSshPublicKeyRequest object.

  Fields:
    parent: The parent project and zone for the signing request. This is
      needed to properly ensure per-organization ISS processing and
      potentially to provide for the possibility of zone-specific certificates
      used in the signing process.
    signSshPublicKeyRequest: A SignSshPublicKeyRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    signSshPublicKeyRequest = _messages.MessageField('SignSshPublicKeyRequest', 2)