from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import custom_target_type_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import manifest_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
def CreateDeliveryPipeline(self, pipeline_config):
    """Creates a delivery pipeline resource.

    Args:
      pipeline_config: apitools.base.protorpclite.messages.Message, delivery
        pipeline message.

    Returns:
      The operation message.
    """
    log.debug('Creating delivery pipeline: ' + repr(pipeline_config))
    return self._pipeline_service.Patch(self.messages.ClouddeployProjectsLocationsDeliveryPipelinesPatchRequest(deliveryPipeline=pipeline_config, allowMissing=True, name=pipeline_config.name, updateMask=manifest_util.PIPELINE_UPDATE_MASK))