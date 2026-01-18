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
def AddClusterDNSFlags(parser, release_track=base.ReleaseTrack.GA, hidden=False):
    """Adds flags related to clusterDNS to parser.

  This includes:
  --cluster-dns={clouddns|kubedns|default},
  --cluster-dns-scope={cluster|vpc},
  --cluster-dns-domain=string
  --additive-vpc-scope-dns-domain=string,
  --disable-additive-vpc-scope

  Args:
    parser: A given parser.
    release_track: Release track the flags are being added to.
    hidden: Indicates that the flags are hidden.
  """
    group = parser.add_argument_group('ClusterDNS', hidden=hidden)
    group.add_argument('--cluster-dns', choices=_DNS_PROVIDER, help='DNS provider to use for this cluster.', hidden=hidden)
    group.add_argument('--cluster-dns-scope', choices=_DNS_SCOPE, help='            DNS scope for the Cloud DNS zone created - valid only with\n             `--cluster-dns=clouddns`. Defaults to cluster.', hidden=hidden)
    group.add_argument('--cluster-dns-domain', help='            DNS domain for this cluster.\n            The default value is `cluster.local`.\n            This is configurable when `--cluster-dns=clouddns` and\n             `--cluster-dns-scope=vpc` are set.\n            The value must be a valid DNS subdomain as defined in RFC 1123.\n            ', hidden=hidden)
    AddAdditiveVPCScopeFlags(group, release_track=release_track, hidden=hidden)