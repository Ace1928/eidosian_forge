from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util.semver import SemVer
def Compare(self, current_master_version, current_cluster_version):
    """Compares the cluster and master versions and returns an enum."""
    if current_master_version == current_cluster_version:
        return self.UP_TO_DATE
    master_version = SemVer(current_master_version)
    cluster_version = SemVer(current_cluster_version)
    major, minor, _ = master_version.Distance(cluster_version)
    if major != 0 or minor > 2:
        return self.UNSUPPORTED
    elif minor > 1:
        return self.SUPPORT_ENDING
    else:
        return self.UPGRADE_AVAILABLE