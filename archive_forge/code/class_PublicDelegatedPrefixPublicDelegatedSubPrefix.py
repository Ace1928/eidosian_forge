from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicDelegatedPrefixPublicDelegatedSubPrefix(_messages.Message):
    """Represents a sub PublicDelegatedPrefix.

  Enums:
    StatusValueValuesEnum: [Output Only] The status of the sub public
      delegated prefix.

  Fields:
    delegateeProject: Name of the project scoping this
      PublicDelegatedSubPrefix.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    ipCidrRange: The IP address range, in CIDR format, represented by this sub
      public delegated prefix.
    isAddress: Whether the sub prefix is delegated to create Address resources
      in the delegatee project.
    name: The name of the sub public delegated prefix.
    region: [Output Only] The region of the sub public delegated prefix if it
      is regional. If absent, the sub prefix is global.
    status: [Output Only] The status of the sub public delegated prefix.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the sub public delegated prefix.

    Values:
      ACTIVE: <no description>
      INACTIVE: <no description>
    """
        ACTIVE = 0
        INACTIVE = 1
    delegateeProject = _messages.StringField(1)
    description = _messages.StringField(2)
    ipCidrRange = _messages.StringField(3)
    isAddress = _messages.BooleanField(4)
    name = _messages.StringField(5)
    region = _messages.StringField(6)
    status = _messages.EnumField('StatusValueValuesEnum', 7)