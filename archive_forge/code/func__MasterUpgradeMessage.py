from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def _MasterUpgradeMessage(name, server_conf, cluster, new_version):
    """Returns the prompt message during a master upgrade.

  Args:
    name: str, the name of the cluster being upgraded.
    server_conf: the server config object.
    cluster: the cluster object.
    new_version: str, the name of the new version, if given.

  Raises:
    NodePoolError: if the node pool name can't be found in the cluster.

  Returns:
    str, a message about which nodes in the cluster will be upgraded and
        to which version.
  """
    if cluster:
        version_message = 'version [{}]'.format(cluster.currentMasterVersion)
    else:
        version_message = 'its current version'
    if not new_version and server_conf:
        new_version = server_conf.defaultClusterVersion
    if new_version:
        new_version_message = 'version [{}]'.format(new_version)
    else:
        new_version_message = 'the default cluster version'
    return 'Master of cluster [{}] will be upgraded from {} to {}.'.format(name, version_message, new_version_message)