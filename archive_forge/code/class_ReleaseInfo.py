from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseInfo(_messages.Message):
    """ReleaseInfo holds extra information about the package release e.g., link
  to an artifact registry oci image.

  Fields:
    ociImagePath: Output only. path to the oci image the service uploads on
      package release creation
  """
    ociImagePath = _messages.StringField(1)