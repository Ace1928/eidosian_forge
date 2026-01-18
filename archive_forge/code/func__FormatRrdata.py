from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def _FormatRrdata(routing_policy_item):
    rrdata = []
    if 'rrdatas' in routing_policy_item:
        rrdata = rrdata + routing_policy_item['rrdatas']
    if 'healthCheckedTargets' in routing_policy_item:
        rrdata = rrdata + ['"{}"'.format(_FormatHealthCheckTarget(target)) for target in routing_policy_item['healthCheckedTargets']['internalLoadBalancers']]
    return ','.join(rrdata)