from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from six.moves import range  # pylint: disable=redefined-builtin
def FetchResourcesAndOutputs(client, messages, project, deployment_name):
    """Returns a ResourcesAndOutputs object for a deployment."""
    try:
        response = client.resources.List(messages.DeploymentmanagerResourcesListRequest(project=project, deployment=deployment_name))
        if response.resources:
            resources = LimitResourcesToDisplay(response.resources)
        else:
            resources = []
        deployment_response = client.deployments.Get(messages.DeploymentmanagerDeploymentsGetRequest(project=project, deployment=deployment_name))
        outputs = []
        manifest = ExtractManifestName(deployment_response)
        if manifest:
            manifest_response = client.manifests.Get(messages.DeploymentmanagerManifestsGetRequest(project=project, deployment=deployment_name, manifest=manifest))
            if manifest_response.layout:
                outputs = FlattenLayoutOutputs(manifest_response.layout)
        return_val = ResourcesAndOutputs(resources, outputs)
        return return_val
    except apitools_exceptions.HttpError as error:
        raise api_exceptions.HttpException(error, HTTP_ERROR_FORMAT)