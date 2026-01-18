from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
def AddParentInfoToAnnotationRequests(ref, args, req):
    """Python hook for yaml commands to wildcard the parent parameter in annotation requests."""
    del ref
    project = utils.GetProject()
    location = utils.GetLocation(args)
    req.parent = utils.GetAnnotationParentTemplate(project, location, args.asset_type, args.asset, args.annotation_set)
    return req