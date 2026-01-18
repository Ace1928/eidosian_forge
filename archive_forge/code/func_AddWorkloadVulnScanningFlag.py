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
def AddWorkloadVulnScanningFlag(parser):
    """Adds Protect Config's Enable Workload Vulnerability Scanning flag to the parser."""
    parser.add_argument('--enable-workload-vulnerability-scanning', default=None, action=actions.DeprecationAction('--enable-workload-vulnerability-scanning', show_message=lambda val: val, warn='`--enable-workload-vulnerability-scanning` is deprecated and will be removed in an upcoming release. Please use `--workload-vulnerability-scanning=standard` to enable workload vulnerability scanning. For more details, please read: https://cloud.google.com/kubernetes-engine/docs/how-to/security-posture-vulnerability-scanning.', action='store_true'), hidden=True, help=textwrap.dedent("      Enables Protect API's workload vulnerability scanning.\n\n      To disable in an existing cluster, explicitly set flag to\n      `--no-enable-workload-vulnerability-scanning`.\n      "))