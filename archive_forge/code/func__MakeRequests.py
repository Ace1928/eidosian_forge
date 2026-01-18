from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.networks.peerings import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _MakeRequests(client, requests, is_async):
    """Helper for making asynchronous or synchronous peering creation requests."""
    if is_async:
        responses, errors = batch_helper.MakeRequests(requests=requests, http=client.apitools_client.http, batch_url=client.batch_url)
        if not errors:
            for operation in responses:
                log.status.write('Creating network peering for [{0}]\n'.format(operation.targetLink))
                log.status.write('Monitor its progress at [{0}]\n'.format(operation.selfLink))
        else:
            utils.RaiseToolException(errors)
    else:
        responses = client.MakeRequests(requests)
    return responses