from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.ai.models import client as model_client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.credentials import requests
from six.moves import http_client
def UndeployModelBeta(self, endpoint_ref, deployed_model_id, traffic_split=None):
    """Undeploys a model from an endpoint using v1beta1 API.

    Args:
      endpoint_ref: Resource, the parsed endpoint that the model is undeployed
        from.
      deployed_model_id: str, Id of the deployed model to be undeployed.
      traffic_split: dict or None, the new traffic split of the endpoint.

    Returns:
      A long-running operation for UndeployModel.
    """
    undeployed_model_req = self.messages.GoogleCloudAiplatformV1beta1UndeployModelRequest(deployedModelId=deployed_model_id)
    if traffic_split is not None:
        additional_properties = []
        for key, value in sorted(traffic_split.items()):
            additional_properties.append(undeployed_model_req.TrafficSplitValue().AdditionalProperty(key=key, value=value))
        undeployed_model_req.trafficSplit = undeployed_model_req.TrafficSplitValue(additionalProperties=additional_properties)
    req = self.messages.AiplatformProjectsLocationsEndpointsUndeployModelRequest(endpoint=endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1UndeployModelRequest=undeployed_model_req)
    return self.client.projects_locations_endpoints.UndeployModel(req)