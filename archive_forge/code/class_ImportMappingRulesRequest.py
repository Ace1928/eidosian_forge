from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportMappingRulesRequest(_messages.Message):
    """Request message for 'ImportMappingRules' request.

  Enums:
    RulesFormatValueValuesEnum: Required. The format of the rules content
      file.

  Fields:
    autoCommit: Required. Should the conversion workspace be committed
      automatically after the import operation.
    rulesFiles: Required. One or more rules files.
    rulesFormat: Required. The format of the rules content file.
  """

    class RulesFormatValueValuesEnum(_messages.Enum):
        """Required. The format of the rules content file.

    Values:
      IMPORT_RULES_FILE_FORMAT_UNSPECIFIED: Unspecified rules format.
      IMPORT_RULES_FILE_FORMAT_HARBOUR_BRIDGE_SESSION_FILE: HarbourBridge
        session file.
      IMPORT_RULES_FILE_FORMAT_ORATOPG_CONFIG_FILE: Ora2Pg configuration file.
    """
        IMPORT_RULES_FILE_FORMAT_UNSPECIFIED = 0
        IMPORT_RULES_FILE_FORMAT_HARBOUR_BRIDGE_SESSION_FILE = 1
        IMPORT_RULES_FILE_FORMAT_ORATOPG_CONFIG_FILE = 2
    autoCommit = _messages.BooleanField(1)
    rulesFiles = _messages.MessageField('RulesFile', 2, repeated=True)
    rulesFormat = _messages.EnumField('RulesFormatValueValuesEnum', 3)