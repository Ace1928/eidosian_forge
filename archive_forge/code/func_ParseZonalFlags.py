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
def ParseZonalFlags(args, resources, message=None):
    """Make Frontend suitable for ZonalLister argument namespace.

  Generated client-side filter is stored to args.filter.

  Args:
    args: The argument namespace of BaseLister.
    resources: resources.Registry, The resource registry
    message: The response resource proto message for the request.

  Returns:
    Frontend initialized with information from BaseLister argument namespace.
    Server-side filter is None.
  """
    frontend = _GetBaseListerFrontendPrototype(args, message=message)
    filter_expr = frontend.filter
    if args.zones:
        filter_expr, scope_set = _TranslateZonesFlag(args, resources, message=message)
    elif args.filter and 'zone' in args.filter:
        scope_set = _TranslateZonesFilters(args, resources)
    else:
        scope_set = AllScopes([resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')], zonal=True, regional=False)
    return _Frontend(filter_expr, frontend.max_results, scope_set)