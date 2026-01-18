from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_security import GetClientInstance
from googlecloudsdk.api_lib.network_security import GetMessagesModule
from googlecloudsdk.core import log
def LogRemoveItemsSuccess(response, args):
    log.status.Print('Items were removed from address group [{}].'.format(args.address_group))
    return response