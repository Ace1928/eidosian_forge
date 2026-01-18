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
def _GetListCommandFrontendPrototype(args, message=None):
    """Make Frontend suitable for ListCommand argument namespace.

  Generated filter is a pair (client-side filter, server-side filter).

  Args:
    args: The argument namespace of ListCommand.
    message: The response resource proto message for the request.

  Returns:
    Frontend initialized with information from ListCommand argument namespace.
    Both client-side and server-side filter is returned.
  """
    filter_expr = flags.RewriteFilter(args, message=message)
    max_results = int(args.page_size) if args.page_size else None
    local_filter, _ = filter_expr
    if args.limit and (max_results is None or max_results > args.limit):
        max_results = args.limit
    if not local_filter:
        max_results = None
    return _Frontend(filter_expr=filter_expr, maxResults=max_results)