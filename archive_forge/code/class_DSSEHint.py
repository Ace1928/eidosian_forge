from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DSSEHint(_messages.Message):
    """This submessage provides human-readable hints about the purpose of the
  authority. Because the name of a note acts as its resource reference, it is
  important to disambiguate the canonical name of the Note (which might be a
  UUID for security purposes) from "readable" names more suitable for debug
  output. Note that these hints should not be used to look up authorities in
  security sensitive contexts, such as when looking up attestations to verify.

  Fields:
    humanReadableName: Required. The human readable name of this attestation
      authority, for example "cloudbuild-prod".
  """
    humanReadableName = _messages.StringField(1)