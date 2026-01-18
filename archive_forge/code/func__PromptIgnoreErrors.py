from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.container.gkemulticloud import operations as op_api_util
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _PromptIgnoreErrors(args, resource_client, resource_ref):
    """Prompt for --ignore-errors flag if the resource is in ERROR or DEGRADED state."""
    resp = resource_client.Get(resource_ref)
    messages = util.GetMessagesModule()
    error_states = [messages.GoogleCloudGkemulticloudV1AttachedCluster.StateValueValuesEnum.ERROR, messages.GoogleCloudGkemulticloudV1AttachedCluster.StateValueValuesEnum.DEGRADED, messages.GoogleCloudGkemulticloudV1AwsCluster.StateValueValuesEnum.ERROR, messages.GoogleCloudGkemulticloudV1AwsCluster.StateValueValuesEnum.DEGRADED, messages.GoogleCloudGkemulticloudV1AwsNodePool.StateValueValuesEnum.ERROR, messages.GoogleCloudGkemulticloudV1AwsNodePool.StateValueValuesEnum.DEGRADED, messages.GoogleCloudGkemulticloudV1AzureCluster.StateValueValuesEnum.ERROR, messages.GoogleCloudGkemulticloudV1AzureCluster.StateValueValuesEnum.DEGRADED, messages.GoogleCloudGkemulticloudV1AzureNodePool.StateValueValuesEnum.ERROR, messages.GoogleCloudGkemulticloudV1AzureNodePool.StateValueValuesEnum.DEGRADED]
    if resp.state not in error_states:
        return
    args.ignore_errors = console_io.PromptContinue(message='Cluster or node pool is in ERROR or DEGRADED state. ' + 'Setting --ignore-errors flag.', throw_if_unattended=True, cancel_on_no=False, default=False)