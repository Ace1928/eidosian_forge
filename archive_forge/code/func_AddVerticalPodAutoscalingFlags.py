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
def AddVerticalPodAutoscalingFlags(parser, hidden=False, experimental=False):
    """Adds vertical pod autoscaling related flags to the parser.

  VerticalPodAutoscaling related flags are:
  --enable-vertical-pod-autoscaling
  --enable-experimental-vertical-pod-autoscaling

  Args:
    parser: A given parser.
    hidden: If true, suppress help text for added options.
    experimental: It true, add experimental vertical pod autoscaling flag
  """
    group = parser.add_group(mutex=True, help='Flags for vertical pod autoscaling:')
    group.add_argument('--enable-vertical-pod-autoscaling', default=None, help='Enable vertical pod autoscaling for a cluster.', hidden=hidden, action='store_true')
    if experimental:
        group.add_argument('--enable-experimental-vertical-pod-autoscaling', default=None, help='Enable experimental vertical pod autoscaling featuresfor a cluster.', hidden=True, action='store_true')