from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetDeleteFlagEnumMap(policy_enum):
    return arg_utils.ChoiceEnumMapper(_DELETE_FLAG_KWARGS['name'], policy_enum, help_str=_DELETE_FLAG_KWARGS['help_str'], default=_DELETE_FLAG_KWARGS['default'])