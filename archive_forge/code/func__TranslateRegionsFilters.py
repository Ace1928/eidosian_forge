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
def _TranslateRegionsFilters(args, resources):
    """Translates simple region=( ...

  ) filters into scope set.

  Args:
    args: The argument namespace of BaseLister.
    resources: resources.Registry, The resource registry

  Returns:
    A region set for the request.
  """
    _, regions = filter_scope_rewriter.FilterScopeRewriter().Rewrite(args.filter, keys={'region'})
    if regions:
        region_list = []
        for r in regions:
            region_resource = resources.Parse(r, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.regions')
            region_list.append(region_resource)
        return RegionSet(region_list)
    return AllScopes([resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')], zonal=False, regional=True)