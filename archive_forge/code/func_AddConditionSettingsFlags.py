from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddConditionSettingsFlags(parser):
    """Adds policy condition flags to the parser."""
    condition_group = parser.add_group(help='        Condition Settings.\n        This will add a condition to the created policy. If any conditions are\n        already specified, this condition will be appended.')
    condition_group.add_argument('--condition-display-name', help='The display name for the condition.')
    condition_group.add_argument('--condition-filter', help='Specifies the "filter" in a metric absence or metric threshold condition.')
    condition_group.add_argument('--aggregation', help='Specifies an Aggregation message as a JSON/YAML value to be applied to the condition. For more information about the format: https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.alertPolicies')
    condition_group.add_argument('--duration', type=arg_parsers.Duration(), help='The duration (e.g. "60s", "2min", etc.) that the condition must hold in order to trigger as true.')
    AddUpdateableConditionFlags(condition_group)