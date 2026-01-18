from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeDescription(_messages.Message):
    """InfoType description.

  Enums:
    SupportedByValueListEntryValuesEnum:

  Fields:
    categories: The category of the infoType.
    description: Description of the infotype. Translated when language is
      provided in the request.
    displayName: Human readable form of the infoType name.
    name: Internal name of the infoType.
    sensitivityScore: The default sensitivity of the infoType.
    supportedBy: Which parts of the API supports this InfoType.
    versions: A list of available versions for the infotype.
  """

    class SupportedByValueListEntryValuesEnum(_messages.Enum):
        """SupportedByValueListEntryValuesEnum enum type.

    Values:
      ENUM_TYPE_UNSPECIFIED: Unused.
      INSPECT: Supported by the inspect operations.
      RISK_ANALYSIS: Supported by the risk analysis operations.
    """
        ENUM_TYPE_UNSPECIFIED = 0
        INSPECT = 1
        RISK_ANALYSIS = 2
    categories = _messages.MessageField('GooglePrivacyDlpV2InfoTypeCategory', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    sensitivityScore = _messages.MessageField('GooglePrivacyDlpV2SensitivityScore', 5)
    supportedBy = _messages.EnumField('SupportedByValueListEntryValuesEnum', 6, repeated=True)
    versions = _messages.MessageField('GooglePrivacyDlpV2VersionDescription', 7, repeated=True)