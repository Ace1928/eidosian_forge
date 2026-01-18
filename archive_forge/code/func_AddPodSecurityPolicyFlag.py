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
def AddPodSecurityPolicyFlag(parser, hidden=False):
    """Adds a --enable-pod-security-policy flag to parser."""
    help_text = 'Enables the pod security policy admission controller for the cluster.  The pod\nsecurity policy admission controller adds fine-grained pod create and update\nauthorization controls through the PodSecurityPolicy API objects. For more\ninformation, see\nhttps://cloud.google.com/kubernetes-engine/docs/how-to/pod-security-policies.\n'
    parser.add_argument('--enable-pod-security-policy', action='store_true', default=None, hidden=hidden, help=help_text)