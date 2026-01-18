from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_security import GetClientInstance
from googlecloudsdk.api_lib.network_security import GetMessagesModule
from googlecloudsdk.core import log
def ListAddressGroupReferences(service, request_type, args):
    address_group = args.CONCEPTS.address_group.Parse()
    request = request_type(addressGroup=address_group.RelativeName())
    return list_pager.YieldFromList(service, request, limit=args.limit, batch_size=args.page_size, method='ListReferences', field='addressGroupReferences', current_token_attribute='pageToken', next_token_attribute='nextPageToken', batch_size_attribute='pageSize')