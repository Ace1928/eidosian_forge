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
def AddMultiScopeListerFlags(parser, zonal=False, regional=False, global_=False):
    """Adds name, --regexp and scope flags as necessary."""
    AddBaseListerArgs(parser)
    scope = parser.add_mutually_exclusive_group()
    if zonal:
        scope.add_argument('--zones', metavar='ZONE', help='If provided, only zonal resources are shown. If arguments are provided, only resources from the given zones are shown.', type=arg_parsers.ArgList())
    if regional:
        scope.add_argument('--regions', metavar='REGION', help='If provided, only regional resources are shown. If arguments are provided, only resources from the given regions are shown.', type=arg_parsers.ArgList())
    if global_:
        scope.add_argument('--global', action='store_true', help='If provided, only global resources are shown.', default=False)