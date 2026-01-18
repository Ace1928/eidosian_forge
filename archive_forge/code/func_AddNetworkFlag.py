from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddNetworkFlag(parser):
    """Adds --network flag."""
    parser.add_argument('--network', help="Existing VPC Network to put the GKE cluster and nodes in. Defaults to 'default' if flag is not provided. If `--subnet=SUBNET` is also specified, subnet must be a subnetwork of the network specified by this `--network=NETWORK` flag.")