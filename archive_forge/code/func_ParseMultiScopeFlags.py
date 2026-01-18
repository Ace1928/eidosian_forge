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
def ParseMultiScopeFlags(args, resources, message=None):
    """Make Frontend suitable for MultiScopeLister argument namespace.

  Generated client-side filter is stored to args.filter.

  Args:
    args: The argument namespace of MultiScopeLister.
    resources: resources.Registry, The resource registry
    message: The response resource proto message for the request.

  Returns:
    Frontend initialized with information from MultiScopeLister argument
    namespace.
  """
    frontend = _GetBaseListerFrontendPrototype(args, message=message)
    filter_expr = frontend.filter
    if getattr(args, 'zones', None):
        filter_expr, scope_set = _TranslateZonesFlag(args, resources, message=message)
    elif args.filter and 'zone' in args.filter:
        scope_set = _TranslateZonesFilters(args, resources)
    elif getattr(args, 'regions', None):
        filter_expr, scope_set = _TranslateRegionsFlag(args, resources, message=message)
    elif args.filter and 'region' in args.filter:
        scope_set = _TranslateRegionsFilters(args, resources)
    elif getattr(args, 'global', None):
        scope_set = GlobalScope([resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')])
        args.filter, filter_expr = flags.RewriteFilter(args, message=message)
    else:
        scope_set = AllScopes([resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')], zonal='zones' in args, regional='regions' in args)
    return _Frontend(filter_expr, frontend.max_results, scope_set)