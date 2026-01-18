from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _UpdateRequestFromArgs(request, args, sql_messages, release_track):
    """Update request with clone options."""
    clone_context = request.instancesCloneRequest.cloneContext
    if args.bin_log_file_name and args.bin_log_position:
        clone_context.binLogCoordinates = sql_messages.BinLogCoordinates(binLogFileName=args.bin_log_file_name, binLogPosition=args.bin_log_position)
    elif args.point_in_time:
        clone_context.pointInTime = args.point_in_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    if args.point_in_time and args.restore_database_name:
        clone_context.databaseNames[:] = [args.restore_database_name]
    if args.point_in_time and args.preferred_zone:
        clone_context.preferredZone = args.preferred_zone
    if release_track == base.ReleaseTrack.ALPHA:
        if args.allocated_ip_range_name:
            clone_context.allocatedIpRange = args.allocated_ip_range_name