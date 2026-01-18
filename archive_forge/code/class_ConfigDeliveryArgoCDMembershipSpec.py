from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigDeliveryArgoCDMembershipSpec(_messages.Message):
    """MembershipSpec defines the ConfigDeliveryArgoCD Feature specification.

  Enums:
    ChannelValueValuesEnum: Channel specifies a channel that can be used to
      resolve a specific addon. Margo will use the same release channel as the
      current cluster. It is not being used. Hidden from customers.

  Fields:
    channel: Channel specifies a channel that can be used to resolve a
      specific addon. Margo will use the same release channel as the current
      cluster. It is not being used. Hidden from customers.
    version: Version specifies the expected ArgoCD version to manage. It is
      only for Ranthos to use/change. Hidden from customers.
  """

    class ChannelValueValuesEnum(_messages.Enum):
        """Channel specifies a channel that can be used to resolve a specific
    addon. Margo will use the same release channel as the current cluster. It
    is not being used. Hidden from customers.

    Values:
      CHANNEL_UNSPECIFIED: CHANNEL_UNSPECIFIED is the default unspecified
        channel field.
      REGULAR: REGULAR refers to access the ConfigDeliveryArgoCD feature
        reasonably soon after they debut, but on a version that has been
        qualified over a longer period of time.
      RAPID: RAPID refers to get the latest ConfigDeliveryArgoCD release as
        early as possible, and be able to use new features the moment they go
        GA.
      STABLE: STABLE refers to prioritize stability over new functionality.
    """
        CHANNEL_UNSPECIFIED = 0
        REGULAR = 1
        RAPID = 2
        STABLE = 3
    channel = _messages.EnumField('ChannelValueValuesEnum', 1)
    version = _messages.StringField(2)