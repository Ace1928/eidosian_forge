from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1RoutineSpec(_messages.Message):
    """Specification that applies to a routine. Valid only for entries with the
  `ROUTINE` type.

  Enums:
    RoutineTypeValueValuesEnum: The type of the routine.

  Fields:
    bigqueryRoutineSpec: Fields specific for BigQuery routines.
    definitionBody: The body of the routine.
    language: The language the routine is written in. The exact value depends
      on the source system. For BigQuery routines, possible values are: *
      `SQL` * `JAVASCRIPT`
    returnType: Return type of the argument. The exact value depends on the
      source system and the language.
    routineArguments: Arguments of the routine.
    routineType: The type of the routine.
  """

    class RoutineTypeValueValuesEnum(_messages.Enum):
        """The type of the routine.

    Values:
      ROUTINE_TYPE_UNSPECIFIED: Unspecified type.
      SCALAR_FUNCTION: Non-builtin permanent scalar function.
      PROCEDURE: Stored procedure.
    """
        ROUTINE_TYPE_UNSPECIFIED = 0
        SCALAR_FUNCTION = 1
        PROCEDURE = 2
    bigqueryRoutineSpec = _messages.MessageField('GoogleCloudDatacatalogV1BigQueryRoutineSpec', 1)
    definitionBody = _messages.StringField(2)
    language = _messages.StringField(3)
    returnType = _messages.StringField(4)
    routineArguments = _messages.MessageField('GoogleCloudDatacatalogV1RoutineSpecArgument', 5, repeated=True)
    routineType = _messages.EnumField('RoutineTypeValueValuesEnum', 6)