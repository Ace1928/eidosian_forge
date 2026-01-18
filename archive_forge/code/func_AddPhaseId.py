from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPhaseId(parser, required=True, hidden=False):
    """Adds phase-id flag."""
    parser.add_argument('--phase-id', hidden=hidden, help='Phase ID on a rollout resource', required=required)