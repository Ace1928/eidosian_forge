from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteSqlRequest(_messages.Message):
    """The request for ExecuteSql and ExecuteStreamingSql.

  Enums:
    QueryModeValueValuesEnum: Used to control the amount of debugging
      information returned in ResultSetStats. If partition_token is set,
      query_mode can only be set to QueryMode.NORMAL.

  Messages:
    ParamTypesValue: It is not always possible for Cloud Spanner to infer the
      right SQL type from a JSON value. For example, values of type `BYTES`
      and values of type `STRING` both appear in params as JSON strings. In
      these cases, `param_types` can be used to specify the exact SQL type for
      some or all of the SQL statement parameters. See the definition of Type
      for more information about SQL types.
    ParamsValue: Parameter names and values that bind to placeholders in the
      SQL string. A parameter placeholder consists of the `@` character
      followed by the parameter name (for example, `@firstName`). Parameter
      names must conform to the naming requirements of identifiers as
      specified at https://cloud.google.com/spanner/docs/lexical#identifiers.
      Parameters can appear anywhere that a literal value is expected. The
      same parameter name can be used more than once, for example: `"WHERE id
      > @msg_id AND id < @msg_id + 100"` It is an error to execute a SQL
      statement with unbound parameters.

  Fields:
    dataBoostEnabled: If this is for a partitioned query and this field is set
      to `true`, the request is executed with Spanner Data Boost independent
      compute resources. If the field is set to `true` but the request does
      not set `partition_token`, the API returns an `INVALID_ARGUMENT` error.
    directedReadOptions: Directed read options for this request.
    paramTypes: It is not always possible for Cloud Spanner to infer the right
      SQL type from a JSON value. For example, values of type `BYTES` and
      values of type `STRING` both appear in params as JSON strings. In these
      cases, `param_types` can be used to specify the exact SQL type for some
      or all of the SQL statement parameters. See the definition of Type for
      more information about SQL types.
    params: Parameter names and values that bind to placeholders in the SQL
      string. A parameter placeholder consists of the `@` character followed
      by the parameter name (for example, `@firstName`). Parameter names must
      conform to the naming requirements of identifiers as specified at
      https://cloud.google.com/spanner/docs/lexical#identifiers. Parameters
      can appear anywhere that a literal value is expected. The same parameter
      name can be used more than once, for example: `"WHERE id > @msg_id AND
      id < @msg_id + 100"` It is an error to execute a SQL statement with
      unbound parameters.
    partitionToken: If present, results will be restricted to the specified
      partition previously created using PartitionQuery(). There must be an
      exact match for the values of fields common to this message and the
      PartitionQueryRequest message used to create this partition_token.
    queryMode: Used to control the amount of debugging information returned in
      ResultSetStats. If partition_token is set, query_mode can only be set to
      QueryMode.NORMAL.
    queryOptions: Query optimizer configuration to use for the given query.
    requestOptions: Common options for this request.
    resumeToken: If this request is resuming a previously interrupted SQL
      statement execution, `resume_token` should be copied from the last
      PartialResultSet yielded before the interruption. Doing this enables the
      new SQL statement execution to resume where the last one left off. The
      rest of the request parameters must exactly match the request that
      yielded this token.
    seqno: A per-transaction sequence number used to identify this request.
      This field makes each request idempotent such that if the request is
      received multiple times, at most one will succeed. The sequence number
      must be monotonically increasing within the transaction. If a request
      arrives for the first time with an out-of-order sequence number, the
      transaction may be aborted. Replays of previously handled requests will
      yield the same response as the first execution. Required for DML
      statements. Ignored for queries.
    sql: Required. The SQL string.
    transaction: The transaction to use. For queries, if none is provided, the
      default is a temporary read-only transaction with strong concurrency.
      Standard DML statements require a read-write transaction. To protect
      against replays, single-use transactions are not supported. The caller
      must either supply an existing transaction ID or begin a new
      transaction. Partitioned DML requires an existing Partitioned DML
      transaction ID.
  """

    class QueryModeValueValuesEnum(_messages.Enum):
        """Used to control the amount of debugging information returned in
    ResultSetStats. If partition_token is set, query_mode can only be set to
    QueryMode.NORMAL.

    Values:
      NORMAL: The default mode. Only the statement results are returned.
      PLAN: This mode returns only the query plan, without any results or
        execution statistics information.
      PROFILE: This mode returns both the query plan and the execution
        statistics along with the results.
    """
        NORMAL = 0
        PLAN = 1
        PROFILE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParamTypesValue(_messages.Message):
        """It is not always possible for Cloud Spanner to infer the right SQL
    type from a JSON value. For example, values of type `BYTES` and values of
    type `STRING` both appear in params as JSON strings. In these cases,
    `param_types` can be used to specify the exact SQL type for some or all of
    the SQL statement parameters. See the definition of Type for more
    information about SQL types.

    Messages:
      AdditionalProperty: An additional property for a ParamTypesValue object.

    Fields:
      additionalProperties: Additional properties of type ParamTypesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParamTypesValue object.

      Fields:
        key: Name of the additional property.
        value: A Type attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Type', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParamsValue(_messages.Message):
        """Parameter names and values that bind to placeholders in the SQL
    string. A parameter placeholder consists of the `@` character followed by
    the parameter name (for example, `@firstName`). Parameter names must
    conform to the naming requirements of identifiers as specified at
    https://cloud.google.com/spanner/docs/lexical#identifiers. Parameters can
    appear anywhere that a literal value is expected. The same parameter name
    can be used more than once, for example: `"WHERE id > @msg_id AND id <
    @msg_id + 100"` It is an error to execute a SQL statement with unbound
    parameters.

    Messages:
      AdditionalProperty: An additional property for a ParamsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataBoostEnabled = _messages.BooleanField(1)
    directedReadOptions = _messages.MessageField('DirectedReadOptions', 2)
    paramTypes = _messages.MessageField('ParamTypesValue', 3)
    params = _messages.MessageField('ParamsValue', 4)
    partitionToken = _messages.BytesField(5)
    queryMode = _messages.EnumField('QueryModeValueValuesEnum', 6)
    queryOptions = _messages.MessageField('QueryOptions', 7)
    requestOptions = _messages.MessageField('RequestOptions', 8)
    resumeToken = _messages.BytesField(9)
    seqno = _messages.IntegerField(10)
    sql = _messages.StringField(11)
    transaction = _messages.MessageField('TransactionSelector', 12)