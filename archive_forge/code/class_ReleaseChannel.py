from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseChannel(_messages.Message):
    """ReleaseChannel indicates which release channel a cluster is subscribed
  to. Release channels are arranged in order of risk. When a cluster is
  subscribed to a release channel, Google maintains both the master version
  and the node version. Node auto-upgrade defaults to true and cannot be
  disabled.

  Enums:
    ChannelValueValuesEnum: channel specifies which release channel the
      cluster is subscribed to.

  Fields:
    channel: channel specifies which release channel the cluster is subscribed
      to.
  """

    class ChannelValueValuesEnum(_messages.Enum):
        """channel specifies which release channel the cluster is subscribed to.

    Values:
      UNSPECIFIED: No channel specified.
      RAPID: RAPID channel is offered on an early access basis for customers
        who want to test new releases. WARNING: Versions available in the
        RAPID Channel may be subject to unresolved issues with no known
        workaround and are not subject to any SLAs.
      REGULAR: Clusters subscribed to REGULAR receive versions that are
        considered GA quality. REGULAR is intended for production users who
        want to take advantage of new features.
      STABLE: Clusters subscribed to STABLE receive versions that are known to
        be stable and reliable in production.
    """
        UNSPECIFIED = 0
        RAPID = 1
        REGULAR = 2
        STABLE = 3
    channel = _messages.EnumField('ChannelValueValuesEnum', 1)