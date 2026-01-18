from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def GetConditionFromArgs(args, messages):
    """Builds a Condition message from args."""
    condition_base_flags = ['--condition-filter', '--condition', '--condition-from-file']
    ValidateAtleastOneSpecified(args, condition_base_flags)
    condition = None
    condition_string = args.condition or args.condition_from_file
    if condition_string:
        condition = MessageFromString(condition_string, messages.Condition, 'Condition')
    aggregations = None
    if args.aggregation:
        aggregations = [MessageFromString(args.aggregation, messages.Aggregation, 'Aggregation')]
    return BuildCondition(messages, condition=condition, display_name=args.condition_display_name, aggregations=aggregations, trigger_count=args.trigger_count, trigger_percent=args.trigger_percent, duration=_FormatDuration(args.duration), condition_filter=args.condition_filter, if_value=args.if_value)