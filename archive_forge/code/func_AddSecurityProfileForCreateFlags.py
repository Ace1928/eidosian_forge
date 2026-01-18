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
def AddSecurityProfileForCreateFlags(parser, hidden=False):
    """Adds flags related to Security Profile to the parser for cluster creation.

  Args:
    parser: A given parser.
    hidden: Whether or not to hide the help text.
  """
    group = parser.add_group(help='Flags for Security Profile:')
    group.add_argument('--security-profile', hidden=hidden, help='Name and version of the security profile to be applied to the cluster.\n\nExamples:\n\n  $ {command} example-cluster --security-profile=default-1.0-gke.0\n')
    group.add_argument('--security-profile-runtime-rules', default=True, action='store_true', hidden=hidden, help='Apply runtime rules in the specified security profile to the cluster.\nWhen enabled (by default), a security profile controller and webhook\nare deployed on the cluster to enforce the runtime rules. If\n--no-security-profile-runtime-rules is specified to disable this\nfeature, only bootstrapping rules are applied, and no security profile\ncontroller or webhook are installed.\n')