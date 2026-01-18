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
def ParseNamesAndRegexpFlags(args, resources, message=None):
    """Makes Frontend suitable for GlobalLister argument namespace.

  Stores generated client-side filter in args.filter.

  Args:
    args: The argument namespace of BaseLister.
    resources: resources.Registry, The resource registry
    message: The resource proto message.

  Returns:
    Frontend initialized with information from BaseLister argument namespace.
  """
    frontend = _GetBaseListerFrontendPrototype(args, message=message)
    scope_set = GlobalScope([resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')])
    args.filter, filter_expr = flags.RewriteFilter(args, message=message)
    return _Frontend(filter_expr, frontend.max_results, scope_set)