from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddUpdateMask(ref, args, req):
    """Hook to add update mask."""
    del ref
    update_mask = []
    if args.IsKnownAndSpecified('level'):
        update_mask.append('access_levels')
    if args.IsKnownAndSpecified('dry_run_level'):
        update_mask.append('dry_run_access_levels')
    if not update_mask:
        raise calliope_exceptions.MinimumArgumentException(['--level', '--dry_run_level'])
    req.updateMask = ','.join(update_mask)
    return req