from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddIpAliasEnvironmentFlags(update_type_group, support_max_pods_per_node):
    """Adds flags related to IP aliasing to parser.

  IP alias flags are related to similar flags found within GKE SDK:
    /third_party/py/googlecloudsdk/command_lib/container/flags.py

  Args:
    update_type_group: argument group, the group to which flag should be added.
    support_max_pods_per_node: bool, if specifying maximum number of pods is
      supported.
  """
    group = update_type_group.add_group(help='IP Alias (VPC-native)')
    ENABLE_IP_ALIAS_FLAG.AddToParser(group)
    CLUSTER_IPV4_CIDR_FLAG.AddToParser(group)
    SERVICES_IPV4_CIDR_FLAG.AddToParser(group)
    CLUSTER_SECONDARY_RANGE_NAME_FLAG.AddToParser(group)
    SERVICES_SECONDARY_RANGE_NAME_FLAG.AddToParser(group)
    ENABLE_IP_MASQ_AGENT_FLAG.AddToParser(group)
    if support_max_pods_per_node:
        MAX_PODS_PER_NODE.AddToParser(group)