from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
class AutoDeleteFlag(enum.Enum):
    """CLI flag values for `auto-delete' flag."""
    NEVER = 'never'
    ON_PERMANENT_INSTANCE_DELETION = 'on-permanent-instance-deletion'

    def GetAutoDeleteEnumValue(self, base_enum):
        return base_enum(self.name)

    @staticmethod
    def ValidateAutoDeleteFlag(flag_value, flag_name):
        values = [auto_delete_flag_value.value for auto_delete_flag_value in AutoDeleteFlag]
        if flag_value not in values:
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='Value for [auto-delete] must be [never] or [on-permanent-instance-deletion], not [{0}]'.format(flag_value))
        return AutoDeleteFlag(flag_value)

    @staticmethod
    def ValidatorWithFlagName(flag_name):

        def Validator(flag_value):
            return AutoDeleteFlag.ValidateAutoDeleteFlag(flag_value, flag_name)
        return Validator