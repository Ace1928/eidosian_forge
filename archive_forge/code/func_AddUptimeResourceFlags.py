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
def AddUptimeResourceFlags(parser):
    """Adds uptime check resource settings flags to the parser."""
    uptime_resource_group = parser.add_group(help='Uptime check resource.', mutex=True, required=True)
    monitored_resource_group = uptime_resource_group.add_group(help='Monitored resource')
    monitored_resource_group.add_argument('--resource-type', help='Type of monitored resource, defaults to `uptime-url`.', choices=UPTIME_MONITORED_RESOURCES)
    base.Argument('--resource-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=str, value_type=str), action=arg_parsers.UpdateAction, required=True, help='Values for all of the labels listed in the associated monitored resource descriptor.\n            See https://cloud.google.com/monitoring/api/resources for more information and allowed\n            keys.').AddToParser(monitored_resource_group)
    group_resource_group = uptime_resource_group.add_group(help='Monitored resource group')
    group_resource_group.add_argument('--group-type', help='The resource type of the group members, defaults to `gce-instance`.', choices=UPTIME_GROUP_RESOURCES)
    group_resource_group.add_argument('--group-id', help='The group of resources being monitored.', required=True, type=str)
    uptime_resource_group.add_argument('--synthetic-target', help='The target of the Synthetic Monitor.\n        This is the fully qualified GCFv2 resource name.\n        ', type=str)