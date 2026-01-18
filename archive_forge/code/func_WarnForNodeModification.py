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
def WarnForNodeModification(args, enable_autorepair):
    if not (args.image_type or '').lower().startswith('ubuntu'):
        return
    if enable_autorepair or args.enable_autoupgrade:
        log.status.Print('Note: Modifications on the boot disks of node VMs do not persist across node recreations. Nodes are recreated during manual-upgrade, auto-upgrade, auto-repair, and auto-scaling. To preserve modifications across node recreation, use a DaemonSet.')