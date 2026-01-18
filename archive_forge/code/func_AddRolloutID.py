from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddRolloutID(parser, hidden=False):
    """Adds rollout-id flag."""
    parser.add_argument('--rollout-id', hidden=hidden, help='ID to assign to the generated rollout for promotion.')