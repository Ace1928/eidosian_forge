from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterPersistenceConfig(_messages.Message):
    """Configuration of the persistence functionality.

  Enums:
    ModeValueValuesEnum: Optional. The mode of persistence.

  Fields:
    aofConfig: Optional. AOF configuration. This field will be ignored if mode
      is not AOF.
    mode: Optional. The mode of persistence.
    rdbConfig: Optional. RDB configuration. This field will be ignored if mode
      is not RDB.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Optional. The mode of persistence.

    Values:
      PERSISTENCE_MODE_UNSPECIFIED: Not set.
      DISABLED: Persistence is disabled, and any snapshot data is deleted.
      RDB: RDB based persistence is enabled.
      AOF: AOF based persistence is enabled.
    """
        PERSISTENCE_MODE_UNSPECIFIED = 0
        DISABLED = 1
        RDB = 2
        AOF = 3
    aofConfig = _messages.MessageField('AOFConfig', 1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)
    rdbConfig = _messages.MessageField('RDBConfig', 3)