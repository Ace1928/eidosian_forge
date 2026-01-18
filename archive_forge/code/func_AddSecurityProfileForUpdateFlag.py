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
def AddSecurityProfileForUpdateFlag(parser, hidden=False):
    """Adds --security-profile to specify security profile for cluster update.

  Args:
    parser: A given parser.
    hidden: Whether or not to hide the help text.
  """
    parser.add_argument('--security-profile', hidden=hidden, help='Name and version of the security profile to be applied to the cluster.\nIf not specified, the current setting of security profile will be\npreserved.\n\nExamples:\n\n  $ {command} example-cluster --security-profile=default-1.0-gke.1\n')