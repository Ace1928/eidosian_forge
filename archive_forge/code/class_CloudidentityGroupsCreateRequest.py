from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsCreateRequest(_messages.Message):
    """A CloudidentityGroupsCreateRequest object.

  Enums:
    InitialGroupConfigValueValuesEnum: Optional. The initial configuration
      option for the `Group`.

  Fields:
    group: A Group resource to be passed as the request body.
    initialGroupConfig: Optional. The initial configuration option for the
      `Group`.
  """

    class InitialGroupConfigValueValuesEnum(_messages.Enum):
        """Optional. The initial configuration option for the `Group`.

    Values:
      INITIAL_GROUP_CONFIG_UNSPECIFIED: Default. Should not be used.
      WITH_INITIAL_OWNER: The end user making the request will be added as the
        initial owner of the `Group`.
      EMPTY: An empty group is created without any initial owners. This can
        only be used by admins of the domain.
    """
        INITIAL_GROUP_CONFIG_UNSPECIFIED = 0
        WITH_INITIAL_OWNER = 1
        EMPTY = 2
    group = _messages.MessageField('Group', 1)
    initialGroupConfig = _messages.EnumField('InitialGroupConfigValueValuesEnum', 2)