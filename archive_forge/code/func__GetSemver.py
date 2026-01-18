from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
from googlecloudsdk.api_lib.container import kubeconfig as kubeconfig_util
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.container.fleet import gateway
from googlecloudsdk.command_lib.container.fleet import gwkubeconfig_util
from googlecloudsdk.command_lib.container.gkemulticloud import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
def _GetSemver(cluster, cluster_id):
    if cluster.controlPlane is None or cluster.controlPlane.version is None:
        raise errors.MissingClusterField(cluster_id, 'version')
    version = cluster.controlPlane.version
    if version.endswith('-next'):
        v = version.replace('-next', '.0', 1)
        return semver.SemVer(v)
    return semver.SemVer(version)