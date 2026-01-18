from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def SetDynamicUserQuery(unused_ref, args, request):
    """Add DynamicGroupUserQuery to DynamicGroupQueries object list.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated dynamic group queries.
  """
    queries = []
    if args.IsSpecified('dynamic_user_query'):
        dg_user_query = args.dynamic_user_query
        version = GetApiVersion(args)
        messages = ci_client.GetMessages(version)
        resource_type = messages.DynamicGroupQuery.ResourceTypeValueValuesEnum
        new_dynamic_group_query = messages.DynamicGroupQuery(resourceType=resource_type.USER, query=dg_user_query)
        queries.append(new_dynamic_group_query)
        dynamic_group_metadata = messages.DynamicGroupMetadata(queries=queries)
        if hasattr(request.group, 'dynamicGroupMetadata'):
            request.group.dynamicGroupMetadata = dynamic_group_metadata
        else:
            request.group = messages.Group(dynamicGroupMetadata=dynamic_group_metadata)
    return request