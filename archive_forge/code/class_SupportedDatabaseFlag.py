from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedDatabaseFlag(_messages.Message):
    """SupportedDatabaseFlag gives general information about a database flag,
  like type and allowed values. This is a static value that is defined on the
  server side, and it cannot be modified by callers. To set the Database flags
  on a particular Instance, a caller should modify the Instance.database_flags
  field.

  Enums:
    SupportedDbVersionsValueListEntryValuesEnum:
    ValueTypeValueValuesEnum:

  Fields:
    acceptsMultipleValues: Whether the database flag accepts multiple values.
      If true, a comma-separated list of stringified values may be specified.
    flagName: The name of the database flag, e.g. "max_allowed_packets". The
      is a possibly key for the Instance.database_flags map field.
    integerRestrictions: Restriction on INTEGER type value.
    name: The name of the flag resource, following Google Cloud conventions,
      e.g.: * projects/{project}/locations/{location}/flags/{flag} This field
      currently has no semantic meaning.
    requiresDbRestart: Whether setting or updating this flag on an Instance
      requires a database restart. If a flag that requires database restart is
      set, the backend will automatically restart the database (making sure to
      satisfy any availability SLO's).
    stringRestrictions: Restriction on STRING type value.
    supportedDbVersions: Major database engine versions for which this flag is
      supported.
    valueType: A ValueTypeValueValuesEnum attribute.
  """

    class SupportedDbVersionsValueListEntryValuesEnum(_messages.Enum):
        """SupportedDbVersionsValueListEntryValuesEnum enum type.

    Values:
      DATABASE_VERSION_UNSPECIFIED: This is an unknown database version.
      POSTGRES_13: DEPRECATED - The database version is Postgres 13.
      POSTGRES_14: The database version is Postgres 14.
      POSTGRES_15: The database version is Postgres 15.
    """
        DATABASE_VERSION_UNSPECIFIED = 0
        POSTGRES_13 = 1
        POSTGRES_14 = 2
        POSTGRES_15 = 3

    class ValueTypeValueValuesEnum(_messages.Enum):
        """ValueTypeValueValuesEnum enum type.

    Values:
      VALUE_TYPE_UNSPECIFIED: This is an unknown flag type.
      STRING: String type flag.
      INTEGER: Integer type flag.
      FLOAT: Float type flag.
      NONE: Denotes that the flag does not accept any values.
    """
        VALUE_TYPE_UNSPECIFIED = 0
        STRING = 1
        INTEGER = 2
        FLOAT = 3
        NONE = 4
    acceptsMultipleValues = _messages.BooleanField(1)
    flagName = _messages.StringField(2)
    integerRestrictions = _messages.MessageField('IntegerRestrictions', 3)
    name = _messages.StringField(4)
    requiresDbRestart = _messages.BooleanField(5)
    stringRestrictions = _messages.MessageField('StringRestrictions', 6)
    supportedDbVersions = _messages.EnumField('SupportedDbVersionsValueListEntryValuesEnum', 7, repeated=True)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 8)