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
def AddBaseListerArgs(parser, hidden=False):
    """Add arguments defined by base_classes.BaseLister."""
    parser.add_argument('names', action=actions.DeprecationAction('names', show_message=bool, warn='Argument `NAME` is deprecated. Use `--filter="name=( \'NAME\' ... )"` instead.'), metavar='NAME', nargs='*', default=[], completer=compute_completers.InstancesCompleter, hidden=hidden, help='If provided, show details for the specified names and/or URIs of resources.')
    parser.add_argument('--regexp', '-r', hidden=hidden, action=actions.DeprecationAction('regexp', warn='Flag `--regexp` is deprecated. Use `--filter="name~\'REGEXP\'"` instead.'), help='        Regular expression to filter the names of the results  on. Any names\n        that do not match the entire regular expression will be filtered out.        ')