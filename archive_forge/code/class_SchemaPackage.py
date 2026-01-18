from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaPackage(_messages.Message):
    """A schema package contains a set of schemas and type definitions.

  Enums:
    SchematizedParsingTypeValueValuesEnum: Determines how messages that fail
      to parse are handled.
    UnexpectedSegmentHandlingValueValuesEnum: Determines how unexpected
      segments (segments not matched to the schema) are handled.

  Fields:
    ignoreMinOccurs: Flag to ignore all min_occurs restrictions in the schema.
      This means that incoming messages can omit any group, segment, field,
      component, or subcomponent.
    schemas: Schema configs that are layered based on their VersionSources
      that match the incoming message. Schema configs present in higher
      indices override those in lower indices with the same message type and
      trigger event if their VersionSources all match an incoming message.
    schematizedParsingType: Determines how messages that fail to parse are
      handled.
    types: Schema type definitions that are layered based on their
      VersionSources that match the incoming message. Type definitions present
      in higher indices override those in lower indices with the same type
      name if their VersionSources all match an incoming message.
    unexpectedSegmentHandling: Determines how unexpected segments (segments
      not matched to the schema) are handled.
  """

    class SchematizedParsingTypeValueValuesEnum(_messages.Enum):
        """Determines how messages that fail to parse are handled.

    Values:
      SCHEMATIZED_PARSING_TYPE_UNSPECIFIED: Unspecified schematized parsing
        type, equivalent to `SOFT_FAIL`.
      SOFT_FAIL: Messages that fail to parse are still stored and ACKed but a
        parser error is stored in place of the schematized data.
      HARD_FAIL: Messages that fail to parse are rejected from
        ingestion/insertion and return an error code.
    """
        SCHEMATIZED_PARSING_TYPE_UNSPECIFIED = 0
        SOFT_FAIL = 1
        HARD_FAIL = 2

    class UnexpectedSegmentHandlingValueValuesEnum(_messages.Enum):
        """Determines how unexpected segments (segments not matched to the
    schema) are handled.

    Values:
      UNEXPECTED_SEGMENT_HANDLING_MODE_UNSPECIFIED: Unspecified handling mode,
        equivalent to FAIL.
      FAIL: Unexpected segments fail to parse and return an error.
      SKIP: Unexpected segments do not fail, but are omitted from the output.
      PARSE: Unexpected segments do not fail, but are parsed in place and
        added to the current group. If a segment has a type definition, it is
        used, otherwise it is parsed as VARIES.
    """
        UNEXPECTED_SEGMENT_HANDLING_MODE_UNSPECIFIED = 0
        FAIL = 1
        SKIP = 2
        PARSE = 3
    ignoreMinOccurs = _messages.BooleanField(1)
    schemas = _messages.MessageField('Hl7SchemaConfig', 2, repeated=True)
    schematizedParsingType = _messages.EnumField('SchematizedParsingTypeValueValuesEnum', 3)
    types = _messages.MessageField('Hl7TypesConfig', 4, repeated=True)
    unexpectedSegmentHandling = _messages.EnumField('UnexpectedSegmentHandlingValueValuesEnum', 5)