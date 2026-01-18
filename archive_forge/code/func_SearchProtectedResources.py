from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def SearchProtectedResources(scope, key_name, resource_types, args):
    client = GetClientInstance()
    request = GetMessagesModule().KmsinventoryOrganizationsProtectedResourcesSearchRequest(scope=scope, cryptoKey=key_name, resourceTypes=resource_types)
    return list_pager.YieldFromList(client.organizations_protectedResources, request, method='Search', limit=args.limit, batch_size_attribute='pageSize', batch_size=args.page_size, field='protectedResources')