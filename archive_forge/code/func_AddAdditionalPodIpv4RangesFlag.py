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
def AddAdditionalPodIpv4RangesFlag(parser):
    """Adds additional pod IPv4 ranges flag to parser."""
    help_text = 'Additional IP address ranges(by name) for pods that need to be added to the cluster.\n\nExamples:\n\n  $ {command} example-cluster --additional-pod-ipv4-ranges=range1,range2\n'
    parser.add_argument('--additional-pod-ipv4-ranges', metavar='NAME', type=arg_parsers.ArgList(min_length=1), help=help_text)