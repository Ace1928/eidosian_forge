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
def AddAlphaClusterFeatureGatesFlags(parser, hidden=True):
    """Adds the --alpha-cluster-feature-gates flag to parser."""
    help_text = 'Allow selectively enable or disable Kubernetes alpha/beta feature gates on alpha cluster.\nAlpha clusters are not covered by the Kubernetes Engine SLA and should not be used for production workloads.'
    parser.add_argument('--alpha-cluster-feature-gates', type=arg_parsers.ArgList(), default=None, metavar='FEATURE=true|false', hidden=hidden, help=help_text)