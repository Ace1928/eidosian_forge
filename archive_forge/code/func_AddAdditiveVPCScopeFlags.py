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
def AddAdditiveVPCScopeFlags(parser, release_track=base.ReleaseTrack.GA, hidden=False):
    """Adds flags related to DNS Additive VPC scope to parser.

  This includes:
  --additive-vpc-scope-dns-domain=string,
  --disable-additive-vpc-scope

  Args:
    parser: A given parser.
    release_track: Release track the flags are being added to.
    hidden: Indicates that the flags are hidden.
  """
    if release_track != base.ReleaseTrack.GA:
        mutex = parser.add_argument_group('ClusterDNS_AdditiveVPCScope_EnabledDisable', hidden=hidden, mutex=True)
        mutex.add_argument('--disable-additive-vpc-scope', default=None, action='store_true', hidden=hidden, help='Disables Additive VPC Scope.')
        mutex.add_argument('--additive-vpc-scope-dns-domain', default=None, hidden=hidden, help='The domain used in Additive VPC scope. Only works with Cluster Scope.')