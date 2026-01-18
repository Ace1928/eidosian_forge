from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStepExtractArchive(_messages.Message):
    """Extracts an archive of the type specified in the specified directory.

  Enums:
    TypeValueValuesEnum: Required. The type of the archive to extract.

  Fields:
    artifactId: Required. The id of the relevant artifact in the recipe.
    destination: Directory to extract archive to. Defaults to `/` on Linux or
      `C:\\` on Windows.
    type: Required. The type of the archive to extract.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the archive to extract.

    Values:
      ARCHIVE_TYPE_UNSPECIFIED: Indicates that the archive type isn't
        specified.
      TAR: Indicates that the archive is a tar archive with no encryption.
      TAR_GZIP: Indicates that the archive is a tar archive with gzip
        encryption.
      TAR_BZIP: Indicates that the archive is a tar archive with bzip
        encryption.
      TAR_LZMA: Indicates that the archive is a tar archive with lzma
        encryption.
      TAR_XZ: Indicates that the archive is a tar archive with xz encryption.
      ZIP: Indicates that the archive is a zip archive.
    """
        ARCHIVE_TYPE_UNSPECIFIED = 0
        TAR = 1
        TAR_GZIP = 2
        TAR_BZIP = 3
        TAR_LZMA = 4
        TAR_XZ = 5
        ZIP = 6
    artifactId = _messages.StringField(1)
    destination = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)