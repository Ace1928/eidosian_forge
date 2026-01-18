from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallLogConfig(_messages.Message):
    """The available logging options for a firewall rule.

  Enums:
    MetadataValueValuesEnum: This field can only be specified for a particular
      firewall rule if logging is enabled for that rule. This field denotes
      whether to include or exclude metadata for firewall logs.

  Fields:
    enable: This field denotes whether to enable logging for a particular
      firewall rule.
    metadata: This field can only be specified for a particular firewall rule
      if logging is enabled for that rule. This field denotes whether to
      include or exclude metadata for firewall logs.
  """

    class MetadataValueValuesEnum(_messages.Enum):
        """This field can only be specified for a particular firewall rule if
    logging is enabled for that rule. This field denotes whether to include or
    exclude metadata for firewall logs.

    Values:
      EXCLUDE_ALL_METADATA: <no description>
      INCLUDE_ALL_METADATA: <no description>
    """
        EXCLUDE_ALL_METADATA = 0
        INCLUDE_ALL_METADATA = 1
    enable = _messages.BooleanField(1)
    metadata = _messages.EnumField('MetadataValueValuesEnum', 2)