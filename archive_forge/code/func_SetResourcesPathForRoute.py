from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def SetResourcesPathForRoute(ref, args, request):
    """Sets the route.network field with a relative resource path.

  Args:
    ref: reference to the route object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    if 'projects/' in args.network:
        return request
    network = resources.REGISTRY.Create('edgenetwork.projects.locations.zones.networks', projectsId=ref.projectsId, locationsId=ref.locationsId, zonesId=ref.zonesId, networksId=args.network)
    request.route.network = network.RelativeName()
    return request