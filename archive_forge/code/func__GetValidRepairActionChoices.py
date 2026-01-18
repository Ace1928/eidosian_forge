from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
@classmethod
def _GetValidRepairActionChoices(cls, dataproc):
    """Get list of valid REPAIR_ACTION values."""
    repair_action_enums = dataproc.messages.NodePool.RepairActionValueValuesEnum
    return [arg_utils.ChoiceToEnumName(n) for n in repair_action_enums.names() if n != 'REPAIR_ACTION_UNSPECIFIED']