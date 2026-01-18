from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def AddDefaultLocationToListRequest(ref, args, req):
    """Python hook for yaml commands to wildcard the region in list requests."""
    del ref
    project = properties.VALUES.core.project.Get(required=True)
    if hasattr(args, 'zone'):
        location = args.region or args.zone or LOCATION_WILDCARD
    else:
        location = args.region or LOCATION_WILDCARD
    req.parent = PARENT_TEMPLATE.format(project, location)
    return req