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
def ProcessResults(resources, field_selector, sort_key_fn=None, reverse_sort=False, limit=None):
    """Process the results from the list query.

  Args:
    resources: The list of returned resources.
    field_selector: Select the primary key for sorting.
    sort_key_fn: Sort the key using this comparison function.
    reverse_sort: Sort the resources in reverse order.
    limit: Limit the number of resourses returned.
  Yields:
    The resource.
  """
    resources = _ConvertProtobufsToDicts(resources)
    if sort_key_fn:
        resources = sorted(resources, key=sort_key_fn, reverse=reverse_sort)
    if limit:
        resources = itertools.islice(resources, limit)
    for resource in resources:
        if field_selector:
            yield field_selector.Apply(resource)
        else:
            yield resource