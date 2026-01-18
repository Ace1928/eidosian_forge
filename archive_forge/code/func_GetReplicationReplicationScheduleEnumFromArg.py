from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetReplicationReplicationScheduleEnumFromArg(choice, messages):
    """Returns the Choice Enum for Replication Schedule.

  Args:
    choice: The choice for replication schedule input as string.
    messages: The messages module.

  Returns:
    The replication schedule enum.
  """
    return arg_utils.ChoiceToEnum(choice=choice, enum_type=messages.Replication.ReplicationScheduleValueValuesEnum)