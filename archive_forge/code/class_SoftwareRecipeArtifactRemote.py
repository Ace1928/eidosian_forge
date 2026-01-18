from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeArtifactRemote(_messages.Message):
    """Specifies an artifact available via some URI.

  Fields:
    checksum: Must be provided if `allow_insecure` is `false`. SHA256 checksum
      in hex format, to compare to the checksum of the artifact. If the
      checksum is not empty and it doesn't match the artifact then the recipe
      installation fails before running any of the steps.
    uri: URI from which to fetch the object. It should contain both the
      protocol and path following the format {protocol}://{location}.
  """
    checksum = _messages.StringField(1)
    uri = _messages.StringField(2)