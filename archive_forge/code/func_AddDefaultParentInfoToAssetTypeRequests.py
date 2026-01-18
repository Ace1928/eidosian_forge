from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.media.asset import utils
def AddDefaultParentInfoToAssetTypeRequests(ref, args, req):
    """Python hook for yaml commands to wildcard the location in asset type requests."""
    del ref
    project = utils.GetProject()
    location = utils.GetLocation(args)
    req.parent = utils.GetParentTemplate(project, location)
    return req