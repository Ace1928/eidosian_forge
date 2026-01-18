from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructVpcConnectivityPatch(disable_vpc_connectivity, network, subnetwork, network_attachment, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for vpc connectivity.

  Used only in Composer 3.

  Args:
    disable_vpc_connectivity: bool or None, defines whether to disable
      connectivity with a user's VPC network.
    network: str or None, the Compute Engine network to which to connect the
      environment specified as relative resource name.
    subnetwork: str or None, the Compute Engine subnetwork to which to connect
      the environment specified as relative resource name.
    network_attachment: str or None, the Compute Engine network attachment that
      is used as PSC Network entry point.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    node_config = messages.NodeConfig()
    config = messages.EnvironmentConfig(nodeConfig=node_config)
    update_mask = None
    if disable_vpc_connectivity:
        update_mask = 'config.node_config.network,config.node_config.subnetwork'
    elif network_attachment:
        update_mask = 'config.node_config.network_attachment'
        node_config.composerNetworkAttachment = network_attachment
    elif network and subnetwork:
        update_mask = 'config.node_config.network,config.node_config.subnetwork'
        node_config.network = network
        node_config.subnetwork = subnetwork
    return (update_mask, messages.Environment(config=config))