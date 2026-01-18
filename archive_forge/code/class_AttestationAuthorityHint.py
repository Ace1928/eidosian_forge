from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttestationAuthorityHint(_messages.Message):
    """This submessage provides human-readable hints about the purpose of the
  AttestationAuthority. Because the name of a Note acts as its resource
  reference, it is important to disambiguate the canonical name of the Note
  (which might be a UUID for security purposes) from "readable" names more
  suitable for debug output. Note that these hints should NOT be used to look
  up AttestationAuthorities in security sensitive contexts, such as when
  looking up Attestations to verify.

  Fields:
    humanReadableName: The human readable name of this Attestation Authority,
      for example "qa".
  """
    humanReadableName = _messages.StringField(1)