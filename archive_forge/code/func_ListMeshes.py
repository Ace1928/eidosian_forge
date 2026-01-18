import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.command_lib.container.fleet import api_util as hubapi_util
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def ListMeshes():
    client = network_services.GetClientInstance()
    return client.projects_locations_meshes.List(client.MESSAGES_MODULE.NetworkservicesProjectsLocationsMeshesListRequest(parent='projects/{}/locations/global'.format(properties.VALUES.core.project.Get())))