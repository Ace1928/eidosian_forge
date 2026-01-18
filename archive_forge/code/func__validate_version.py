from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkeonprem import operations
from googlecloudsdk.api_lib.container.gkeonprem import vmware_admin_clusters
from googlecloudsdk.api_lib.container.gkeonprem import vmware_clusters
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.gkeonprem import flags as common_flags
from googlecloudsdk.command_lib.container.vmware import constants
from googlecloudsdk.command_lib.container.vmware import errors
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import semver
def _validate_version(self, cluster, cluster_ref):
    if cluster.onPremVersion is None:
        raise errors.MissingClusterField(cluster_ref.RelativeName(), 'onPremVersion')
    if semver.SemVer(cluster.onPremVersion) < semver.SemVer('1.13.0-gke.1'):
        raise errors.UnsupportedClusterVersion('Central upgrade is only supported in cluster version 1.13.0 and newer. Cluster is at version {}.'.format(cluster.onPremVersion))