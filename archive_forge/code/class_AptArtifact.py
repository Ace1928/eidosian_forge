from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AptArtifact(_messages.Message):
    """A detailed representation of an Apt artifact. Information in the record
  is derived from the archive's control file. See
  https://www.debian.org/doc/debian-policy/ch-controlfields.html

  Enums:
    PackageTypeValueValuesEnum: Output only. An artifact is a binary or source
      package.

  Fields:
    architecture: Output only. Operating system architecture of the artifact.
    component: Output only. Repository component of the artifact.
    controlFile: Output only. Contents of the artifact's control metadata
      file.
    name: Output only. The Artifact Registry resource name of the artifact.
    packageName: Output only. The Apt package name of the artifact.
    packageType: Output only. An artifact is a binary or source package.
  """

    class PackageTypeValueValuesEnum(_messages.Enum):
        """Output only. An artifact is a binary or source package.

    Values:
      PACKAGE_TYPE_UNSPECIFIED: Package type is not specified.
      BINARY: Binary package.
      SOURCE: Source package.
    """
        PACKAGE_TYPE_UNSPECIFIED = 0
        BINARY = 1
        SOURCE = 2
    architecture = _messages.StringField(1)
    component = _messages.StringField(2)
    controlFile = _messages.BytesField(3)
    name = _messages.StringField(4)
    packageName = _messages.StringField(5)
    packageType = _messages.EnumField('PackageTypeValueValuesEnum', 6)