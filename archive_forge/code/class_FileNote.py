from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileNote(_messages.Message):
    """FileNote represents an SPDX File Information section:
  https://spdx.github.io/spdx-spec/4-file-information/

  Enums:
    FileTypeValueValuesEnum: This field provides information about the type of
      file identified

  Fields:
    checksum: Provide a unique identifier to match analysis information on
      each specific file in a package
    fileType: This field provides information about the type of file
      identified
    title: Identify the full path and filename that corresponds to the file
      information in this section
  """

    class FileTypeValueValuesEnum(_messages.Enum):
        """This field provides information about the type of file identified

    Values:
      FILE_TYPE_UNSPECIFIED: Unspecified
      SOURCE: The file is human readable source code (.c, .html, etc.)
      BINARY: The file is a compiled object, target image or binary executable
        (.o, .a, etc.)
      ARCHIVE: The file represents an archive (.tar, .jar, etc.)
      APPLICATION: The file is associated with a specific application type
        (MIME type of application/*)
      AUDIO: The file is associated with an audio file (MIME type of audio/* ,
        e.g. .mp3)
      IMAGE: The file is associated with an picture image file (MIME type of
        image/*, e.g., .jpg, .gif)
      TEXT: The file is human readable text file (MIME type of text/*)
      VIDEO: The file is associated with a video file type (MIME type of
        video/*)
      DOCUMENTATION: The file serves as documentation
      SPDX: The file is an SPDX document
      OTHER: The file doesn't fit into the above categories (generated
        artifacts, data files, etc.)
    """
        FILE_TYPE_UNSPECIFIED = 0
        SOURCE = 1
        BINARY = 2
        ARCHIVE = 3
        APPLICATION = 4
        AUDIO = 5
        IMAGE = 6
        TEXT = 7
        VIDEO = 8
        DOCUMENTATION = 9
        SPDX = 10
        OTHER = 11
    checksum = _messages.StringField(1, repeated=True)
    fileType = _messages.EnumField('FileTypeValueValuesEnum', 2)
    title = _messages.StringField(3)