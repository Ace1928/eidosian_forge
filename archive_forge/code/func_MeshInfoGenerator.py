import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.command_lib.container.fleet import api_util as hubapi_util
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def MeshInfoGenerator(args):
    """Generate meshName from membership, location and project."""
    target_mesh_name = ''
    target_project_number = ''
    meshes = ListMeshes()
    for mesh_info in meshes.meshes:
        if mesh_info.description is None:
            continue
        matcher = re.match('.*projects/(.*)/locations/(.*)/memberships/(.*): ', mesh_info.description)
        if matcher is None:
            continue
        if matcher.group(2) != args.location or matcher.group(3) != args.membership:
            continue
        else:
            matcher_new = re.match('.+/meshes/(.*)', mesh_info.name)
            if matcher_new is None:
                continue
            target_mesh_name = matcher_new.group(1)
            target_project_number = matcher.group(1)
            break
    return (target_mesh_name, target_project_number)