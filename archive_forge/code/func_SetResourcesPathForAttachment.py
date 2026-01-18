from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def SetResourcesPathForAttachment(ref, args, request):
    """Sets the interconnectAttachment.router and interconnectAttachment.interconnect field with a relative resource path.

  Args:
    ref: reference to the interconnectAttachment object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    if 'projects/' not in args.interconnect:
        interconnect = resources.REGISTRY.Create('edgenetwork.projects.locations.zones.interconnects', projectsId=ref.projectsId, locationsId=ref.locationsId, zonesId=ref.zonesId, interconnectsId=args.interconnect)
        request.interconnectAttachment.interconnect = interconnect.RelativeName()
    if args.network and 'projects/' not in args.network:
        network = resources.REGISTRY.Create('edgenetwork.projects.locations.zones.networks', projectsId=ref.projectsId, locationsId=ref.locationsId, zonesId=ref.zonesId, networksId=args.network)
        request.interconnectAttachment.network = network.RelativeName()
    return request