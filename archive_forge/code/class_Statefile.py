from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Statefile(_messages.Message):
    """Contains info about a Terraform state file

  Fields:
    signedUri: Output only. Cloud Storage signed URI used for downloading or
      uploading the state file.
  """
    signedUri = _messages.StringField(1)