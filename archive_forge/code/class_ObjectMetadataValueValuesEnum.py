from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectMetadataValueValuesEnum(_messages.Enum):
    """Optional. ObjectMetadata is used to create Object Tables. Object
    Tables contain a listing of objects (with their metadata) found at the
    source_uris. If ObjectMetadata is set, source_format should be omitted.
    Currently SIMPLE is the only supported Object Metadata type.

    Values:
      OBJECT_METADATA_UNSPECIFIED: Unspecified by default.
      DIRECTORY: A synonym for `SIMPLE`.
      SIMPLE: Directory listing of objects.
    """
    OBJECT_METADATA_UNSPECIFIED = 0
    DIRECTORY = 1
    SIMPLE = 2