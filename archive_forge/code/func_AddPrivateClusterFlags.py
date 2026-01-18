from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddPrivateClusterFlags(parser, default=None, with_deprecated=False):
    """Adds flags related to private clusters to parser."""
    default = {} if default is None else default
    group = parser.add_argument_group('Private Clusters')
    if with_deprecated:
        if 'private_cluster' not in default:
            group.add_argument('--private-cluster', help='Cluster is created with no public IP addresses on the cluster nodes.', default=None, action=actions.DeprecationAction('private-cluster', warn='The --private-cluster flag is deprecated and will be removed in a future release. Use --enable-private-nodes instead.', action='store_true'))
    if 'enable_private_nodes' not in default:
        group.add_argument('--enable-private-nodes', help='Cluster is created with no public IP addresses on the cluster nodes.', default=None, action='store_true')
    if 'enable_private_endpoint' not in default:
        group.add_argument('--enable-private-endpoint', help='Cluster is managed using the private IP address of the master API endpoint.', default=None, action='store_true')
    if 'master_ipv4_cidr' not in default:
        group.add_argument('--master-ipv4-cidr', help='IPv4 CIDR range to use for the master network.  This should have a netmask of size /28 and should be used in conjunction with the --enable-private-nodes flag.', default=None)