from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def ParseResponsePolicyNetworks(value, project, version):
    """Build a list of ResponsePolicyNetworks from command line args."""
    networks = ParseNetworks(value, project, version)
    return ResponsePolicyNetworkProcessor(networks, version)