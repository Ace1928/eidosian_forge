from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
def AddInstanceNameToRequest(ref, args, req):
    """Python hook for yaml commands to process the source instance name."""
    del ref
    project = properties.VALUES.core.project.Get(required=True)
    req.snapshot.sourceInstance = INSTANCE_NAME_TEMPLATE.format(project, args.instance_zone, args.instance)
    return req