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
def GetRegionalResourcesDicts(service, project, requested_regions, filter_expr, http, batch_url, errors):
    """Lists resources that are scoped by region and returns them as dicts.

  Args:
    service: An apitools service object.
    project: The Compute Engine project name for which listing should be
      performed.
    requested_regions: A list of region names that can be used to
      control the scope of the list call.
    filter_expr: A filter to pass to the list API calls.
    http: An httplib2.Http-like object.
    batch_url: The handler for making batch requests.
    errors: A list for capturing errors.

  Returns:
    A list of dicts representing the results.
  """
    return _GetResources(service=service, project=project, scopes=requested_regions, scope_name='region', filter_expr=filter_expr, http=http, batch_url=batch_url, errors=errors, make_requests=request_helper.ListJson)