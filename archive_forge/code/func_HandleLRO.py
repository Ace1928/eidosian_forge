from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pprint
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def HandleLRO(client, result_operation, service, no_resource=False):
    """Uses the waiter library to handle LRO synchronous execution."""
    op_resource = resources.REGISTRY.ParseRelativeName(result_operation.name, collection='datamigration.projects.locations.operations')
    if no_resource:
        poller = waiter.CloudOperationPollerNoResources(client.projects_locations_operations)
    else:
        poller = CloudDmsOperationPoller(service, client.projects_locations_operations)
    try:
        waiter.WaitFor(poller, op_resource, 'Waiting for operation [{}] to complete'.format(result_operation.name))
    except waiter.TimeoutError:
        log.status.Print('The operations may still be underway remotely and may still succeed. You may check the operation status for the following operation  [{}]'.format(result_operation.name))
        return