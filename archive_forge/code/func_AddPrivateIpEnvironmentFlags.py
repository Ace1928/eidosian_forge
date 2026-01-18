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
def AddPrivateIpEnvironmentFlags(update_type_group, release_track):
    """Adds flags related to private clusters to parser.

  Private cluster flags are related to similar flags found within GKE SDK:
    /third_party/py/googlecloudsdk/command_lib/container/flags.py

  Args:
    update_type_group: argument group, the group to which flag should be added.
    release_track: which release track messages should we use.
  """
    group = update_type_group.add_group(help='Private Clusters')
    ENABLE_PRIVATE_ENVIRONMENT_FLAG.AddToParser(group)
    ENABLE_PRIVATE_ENDPOINT_FLAG.AddToParser(group)
    MASTER_IPV4_CIDR_FLAG.AddToParser(group)
    WEB_SERVER_IPV4_CIDR_FLAG.AddToParser(group)
    CLOUD_SQL_IPV4_CIDR_FLAG.AddToParser(group)
    COMPOSER_NETWORK_IPV4_CIDR_FLAG.AddToParser(group)
    CONNECTION_SUBNETWORK_FLAG.AddToParser(group)
    if release_track == base.ReleaseTrack.GA:
        CONNECTION_TYPE_FLAG_GA.choice_arg.AddToParser(group)
    elif release_track == base.ReleaseTrack.BETA:
        CONNECTION_TYPE_FLAG_BETA.choice_arg.AddToParser(group)
    elif release_track == base.ReleaseTrack.ALPHA:
        CONNECTION_TYPE_FLAG_ALPHA.choice_arg.AddToParser(group)
    ENABLE_PRIVATELY_USED_PUBLIC_IPS_FLAG.AddToParser(group)