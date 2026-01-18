from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DataCollectorConfig(_messages.Message):
    """Data collector and its configuration.

  Enums:
    TypeValueValuesEnum: Data type accepted by the data collector.

  Fields:
    name: Name of the data collector in the following format:
      `organizations/{org}/datacollectors/{datacollector}`
    type: Data type accepted by the data collector.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Data type accepted by the data collector.

    Values:
      TYPE_UNSPECIFIED: For future compatibility.
      INTEGER: For integer values.
      FLOAT: For float values.
      STRING: For string values.
      BOOLEAN: For boolean values.
      DATETIME: For datetime values.
    """
        TYPE_UNSPECIFIED = 0
        INTEGER = 1
        FLOAT = 2
        STRING = 3
        BOOLEAN = 4
        DATETIME = 5
    name = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)