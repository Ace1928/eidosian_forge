from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Routine(_messages.Message):
    """A user-defined function or a stored procedure.

  Enums:
    DataGovernanceTypeValueValuesEnum: Optional. If set to `DATA_MASKING`, the
      function is validated and made available as a masking function. For more
      information, see [Create custom masking
      routines](https://cloud.google.com/bigquery/docs/user-defined-
      functions#custom-mask).
    DeterminismLevelValueValuesEnum: Optional. The determinism level of the
      JavaScript UDF, if defined.
    LanguageValueValuesEnum: Optional. Defaults to "SQL" if
      remote_function_options field is absent, not set otherwise.
    RoutineTypeValueValuesEnum: Required. The type of routine.
    SecurityModeValueValuesEnum: Optional. The security mode of the routine,
      if defined. If not defined, the security mode is automatically
      determined from the routine's configuration.

  Fields:
    arguments: Optional.
    creationTime: Output only. The time when this routine was created, in
      milliseconds since the epoch.
    dataGovernanceType: Optional. If set to `DATA_MASKING`, the function is
      validated and made available as a masking function. For more
      information, see [Create custom masking
      routines](https://cloud.google.com/bigquery/docs/user-defined-
      functions#custom-mask).
    definitionBody: Required. The body of the routine. For functions, this is
      the expression in the AS clause. If language=SQL, it is the substring
      inside (but excluding) the parentheses. For example, for the function
      created with the following statement: `CREATE FUNCTION JoinLines(x
      string, y string) as (concat(x, "\\n", y))` The definition_body is
      `concat(x, "\\n", y)` (\\n is not replaced with linebreak). If
      language=JAVASCRIPT, it is the evaluated string in the AS clause. For
      example, for the function created with the following statement: `CREATE
      FUNCTION f() RETURNS STRING LANGUAGE js AS 'return "\\n";\\n'` The
      definition_body is `return "\\n";\\n` Note that both \\n are replaced with
      linebreaks.
    description: Optional. The description of the routine, if defined.
    determinismLevel: Optional. The determinism level of the JavaScript UDF,
      if defined.
    etag: Output only. A hash of this resource.
    importedLibraries: Optional. If language = "JAVASCRIPT", this field stores
      the path of the imported JAVASCRIPT libraries.
    language: Optional. Defaults to "SQL" if remote_function_options field is
      absent, not set otherwise.
    lastModifiedTime: Output only. The time when this routine was last
      modified, in milliseconds since the epoch.
    remoteFunctionOptions: Optional. Remote function specific options.
    returnTableType: Optional. Can be set only if routine_type =
      "TABLE_VALUED_FUNCTION". If absent, the return table type is inferred
      from definition_body at query time in each query that references this
      routine. If present, then the columns in the evaluated table result will
      be cast to match the column types specified in return table type, at
      query time.
    returnType: Optional if language = "SQL"; required otherwise. Cannot be
      set if routine_type = "TABLE_VALUED_FUNCTION". If absent, the return
      type is inferred from definition_body at query time in each query that
      references this routine. If present, then the evaluated result will be
      cast to the specified returned type at query time. For example, for the
      functions created with the following statements: * `CREATE FUNCTION
      Add(x FLOAT64, y FLOAT64) RETURNS FLOAT64 AS (x + y);` * `CREATE
      FUNCTION Increment(x FLOAT64) AS (Add(x, 1));` * `CREATE FUNCTION
      Decrement(x FLOAT64) RETURNS FLOAT64 AS (Add(x, -1));` The return_type
      is `{type_kind: "FLOAT64"}` for `Add` and `Decrement`, and is absent for
      `Increment` (inferred as FLOAT64 at query time). Suppose the function
      `Add` is replaced by `CREATE OR REPLACE FUNCTION Add(x INT64, y INT64)
      AS (x + y);` Then the inferred return type of `Increment` is
      automatically changed to INT64 at query time, while the return type of
      `Decrement` remains FLOAT64.
    routineReference: Required. Reference describing the ID of this routine.
    routineType: Required. The type of routine.
    securityMode: Optional. The security mode of the routine, if defined. If
      not defined, the security mode is automatically determined from the
      routine's configuration.
    sparkOptions: Optional. Spark specific options.
    strictMode: Optional. Use this option to catch many common errors. Error
      checking is not exhaustive, and successfully creating a procedure
      doesn't guarantee that the procedure will successfully execute at
      runtime. If `strictMode` is set to `TRUE`, the procedure body is further
      checked for errors such as non-existent tables or columns. The `CREATE
      PROCEDURE` statement fails if the body fails any of these checks. If
      `strictMode` is set to `FALSE`, the procedure body is checked only for
      syntax. For procedures that invoke themselves recursively, specify
      `strictMode=FALSE` to avoid non-existent procedure errors during
      validation. Default value is `TRUE`.
  """

    class DataGovernanceTypeValueValuesEnum(_messages.Enum):
        """Optional. If set to `DATA_MASKING`, the function is validated and made
    available as a masking function. For more information, see [Create custom
    masking routines](https://cloud.google.com/bigquery/docs/user-defined-
    functions#custom-mask).

    Values:
      DATA_GOVERNANCE_TYPE_UNSPECIFIED: The data governance type is
        unspecified.
      DATA_MASKING: The data governance type is data masking.
    """
        DATA_GOVERNANCE_TYPE_UNSPECIFIED = 0
        DATA_MASKING = 1

    class DeterminismLevelValueValuesEnum(_messages.Enum):
        """Optional. The determinism level of the JavaScript UDF, if defined.

    Values:
      DETERMINISM_LEVEL_UNSPECIFIED: The determinism of the UDF is
        unspecified.
      DETERMINISTIC: The UDF is deterministic, meaning that 2 function calls
        with the same inputs always produce the same result, even across 2
        query runs.
      NOT_DETERMINISTIC: The UDF is not deterministic.
    """
        DETERMINISM_LEVEL_UNSPECIFIED = 0
        DETERMINISTIC = 1
        NOT_DETERMINISTIC = 2

    class LanguageValueValuesEnum(_messages.Enum):
        """Optional. Defaults to "SQL" if remote_function_options field is
    absent, not set otherwise.

    Values:
      LANGUAGE_UNSPECIFIED: Default value.
      SQL: SQL language.
      JAVASCRIPT: JavaScript language.
      PYTHON: Python language.
      JAVA: Java language.
      SCALA: Scala language.
    """
        LANGUAGE_UNSPECIFIED = 0
        SQL = 1
        JAVASCRIPT = 2
        PYTHON = 3
        JAVA = 4
        SCALA = 5

    class RoutineTypeValueValuesEnum(_messages.Enum):
        """Required. The type of routine.

    Values:
      ROUTINE_TYPE_UNSPECIFIED: Default value.
      SCALAR_FUNCTION: Non-built-in persistent scalar function.
      PROCEDURE: Stored procedure.
      TABLE_VALUED_FUNCTION: Non-built-in persistent TVF.
      AGGREGATE_FUNCTION: Non-built-in persistent aggregate function.
    """
        ROUTINE_TYPE_UNSPECIFIED = 0
        SCALAR_FUNCTION = 1
        PROCEDURE = 2
        TABLE_VALUED_FUNCTION = 3
        AGGREGATE_FUNCTION = 4

    class SecurityModeValueValuesEnum(_messages.Enum):
        """Optional. The security mode of the routine, if defined. If not
    defined, the security mode is automatically determined from the routine's
    configuration.

    Values:
      SECURITY_MODE_UNSPECIFIED: The security mode of the routine is
        unspecified.
      DEFINER: The routine is to be executed with the privileges of the user
        who defines it.
      INVOKER: The routine is to be executed with the privileges of the user
        who invokes it.
    """
        SECURITY_MODE_UNSPECIFIED = 0
        DEFINER = 1
        INVOKER = 2
    arguments = _messages.MessageField('Argument', 1, repeated=True)
    creationTime = _messages.IntegerField(2)
    dataGovernanceType = _messages.EnumField('DataGovernanceTypeValueValuesEnum', 3)
    definitionBody = _messages.StringField(4)
    description = _messages.StringField(5)
    determinismLevel = _messages.EnumField('DeterminismLevelValueValuesEnum', 6)
    etag = _messages.StringField(7)
    importedLibraries = _messages.StringField(8, repeated=True)
    language = _messages.EnumField('LanguageValueValuesEnum', 9)
    lastModifiedTime = _messages.IntegerField(10)
    remoteFunctionOptions = _messages.MessageField('RemoteFunctionOptions', 11)
    returnTableType = _messages.MessageField('StandardSqlTableType', 12)
    returnType = _messages.MessageField('StandardSqlDataType', 13)
    routineReference = _messages.MessageField('RoutineReference', 14)
    routineType = _messages.EnumField('RoutineTypeValueValuesEnum', 15)
    securityMode = _messages.EnumField('SecurityModeValueValuesEnum', 16)
    sparkOptions = _messages.MessageField('SparkOptions', 17)
    strictMode = _messages.BooleanField(18)