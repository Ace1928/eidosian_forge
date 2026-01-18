from googlecloudsdk.command_lib.network_management.simulation import util
from googlecloudsdk.core import properties
def SetProjectAsParent(unused_ref, unused_args, request):
    """Add parent path to request, since it isn't automatically populated by apitools."""
    project = properties.VALUES.core.project.Get()
    if project is None:
        raise ValueError('Required field project not provided')
    request.parent = 'projects/' + project + '/locations/global'
    return request