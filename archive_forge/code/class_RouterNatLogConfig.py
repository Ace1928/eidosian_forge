from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterNatLogConfig(_messages.Message):
    """Configuration of logging on a NAT.

  Enums:
    FilterValueValuesEnum: Specify the desired filtering of logs on this NAT.
      If unspecified, logs are exported for all connections handled by this
      NAT. This option can take one of the following values: - ERRORS_ONLY:
      Export logs only for connection failures. - TRANSLATIONS_ONLY: Export
      logs only for successful connections. - ALL: Export logs for all
      connections, successful and unsuccessful.

  Fields:
    enable: Indicates whether or not to export logs. This is false by default.
    filter: Specify the desired filtering of logs on this NAT. If unspecified,
      logs are exported for all connections handled by this NAT. This option
      can take one of the following values: - ERRORS_ONLY: Export logs only
      for connection failures. - TRANSLATIONS_ONLY: Export logs only for
      successful connections. - ALL: Export logs for all connections,
      successful and unsuccessful.
  """

    class FilterValueValuesEnum(_messages.Enum):
        """Specify the desired filtering of logs on this NAT. If unspecified,
    logs are exported for all connections handled by this NAT. This option can
    take one of the following values: - ERRORS_ONLY: Export logs only for
    connection failures. - TRANSLATIONS_ONLY: Export logs only for successful
    connections. - ALL: Export logs for all connections, successful and
    unsuccessful.

    Values:
      ALL: Export logs for all (successful and unsuccessful) connections.
      ERRORS_ONLY: Export logs for connection failures only.
      TRANSLATIONS_ONLY: Export logs for successful connections only.
    """
        ALL = 0
        ERRORS_ONLY = 1
        TRANSLATIONS_ONLY = 2
    enable = _messages.BooleanField(1)
    filter = _messages.EnumField('FilterValueValuesEnum', 2)