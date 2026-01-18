from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator, Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.api_lib.container.vmware import version_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _enable_control_plane_v2(self, args: parser_extensions.Namespace):
    """While creating a 1.15+ user cluster, default enable_control_plane_v2 to True if not set."""
    if 'enable_control_plane_v2' in args.GetSpecifiedArgsDict():
        return True
    if 'disable_control_plane_v2' in args.GetSpecifiedArgsDict():
        return False
    default_enable_control_plane_v2 = '1.15.0-gke.0'
    if args.command_path[-1] == 'create' and version_util.Version(args.version).feature_available(default_enable_control_plane_v2):
        return True
    return None