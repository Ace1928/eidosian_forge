from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LookupConfigsRequest(_messages.Message):
    """A request message for getting the configs assigned to the instance.

  Enums:
    ConfigTypesValueListEntryValuesEnum:

  Fields:
    configTypes: Types of configuration system the instance is using. Only
      configs relevant to these configuration types will be returned.
    osInfo: Optional. OS info about the instance that can be used to filter
      its configs. If none is provided, the API will return the configs for
      this instance regardless of its OS.
  """

    class ConfigTypesValueListEntryValuesEnum(_messages.Enum):
        """ConfigTypesValueListEntryValuesEnum enum type.

    Values:
      CONFIG_TYPE_UNSPECIFIED: <no description>
      APT: <no description>
      YUM: <no description>
      GOO: <no description>
      WINDOWS_UPDATE: <no description>
      ZYPPER: <no description>
    """
        CONFIG_TYPE_UNSPECIFIED = 0
        APT = 1
        YUM = 2
        GOO = 3
        WINDOWS_UPDATE = 4
        ZYPPER = 5
    configTypes = _messages.EnumField('ConfigTypesValueListEntryValuesEnum', 1, repeated=True)
    osInfo = _messages.MessageField('LookupConfigsRequestOsInfo', 2)