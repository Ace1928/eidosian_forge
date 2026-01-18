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
def AddSecurityPostureEnumFlag(parser):
    """Adds Kubernetes Security Posture's enablement flag to the parser."""
    parser.add_argument('--security-posture', choices=['disabled', 'standard', 'enterprise'], default=None, hidden=False, help=textwrap.dedent("      Sets the mode of the Kubernetes security posture API's off-cluster features.\n\n      To enable advanced mode explicitly set the flag to\n      `--security-posture=enterprise`.\n\n      To enable in standard mode explicitly set the flag to\n      `--security-posture=standard`\n\n      To disable in an existing cluster, explicitly set the flag to\n      `--security-posture=disabled`.\n      "))