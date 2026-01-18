from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import filter_scope_rewriter
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_projector
import six
def AddZonalListerArgs(parser, hidden=False):
    """Add arguments defined by base_classes.ZonalLister."""
    AddBaseListerArgs(parser, hidden)
    parser.add_argument('--zones', metavar='ZONE', help='If provided, only resources from the given zones are queried.', hidden=hidden, type=arg_parsers.ArgList(min_length=1), completer=compute_completers.ZonesCompleter, default=[])