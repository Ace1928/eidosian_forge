from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ContentSqlScript(_messages.Message):
    """Configuration for the Sql Script content.

  Enums:
    EngineValueValuesEnum: Required. Query Engine to be used for the Sql
      Query.

  Fields:
    engine: Required. Query Engine to be used for the Sql Query.
  """

    class EngineValueValuesEnum(_messages.Enum):
        """Required. Query Engine to be used for the Sql Query.

    Values:
      QUERY_ENGINE_UNSPECIFIED: Value was unspecified.
      SPARK: Spark SQL Query.
    """
        QUERY_ENGINE_UNSPECIFIED = 0
        SPARK = 1
    engine = _messages.EnumField('EngineValueValuesEnum', 1)