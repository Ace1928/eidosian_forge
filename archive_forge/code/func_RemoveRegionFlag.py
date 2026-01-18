from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def RemoveRegionFlag(unused_domain_ref, args, patch_request):
    """Removes region from domain."""
    if args.IsSpecified('remove_region'):
        locs = [loc for loc in patch_request.domain.locations if loc != args.remove_region]
        locs = sorted(set(locs))
        if not locs:
            raise exceptions.ActiveDirectoryError('Cannot remove all regions')
        patch_request.domain.locations = locs
        AddFieldToUpdateMask('locations', patch_request)
    return patch_request