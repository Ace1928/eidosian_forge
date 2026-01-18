from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def ExtractGracefulShutdownFromArgs(args, support_graceful_shutdown=False):
    """Extracts graceful shutdown from args."""
    graceful_shutdown = None
    if support_graceful_shutdown:
        if hasattr(args, 'graceful_shutdown') and args.IsSpecified('graceful_shutdown'):
            graceful_shutdown = {'enabled': args.graceful_shutdown}
        if hasattr(args, 'graceful_shutdown_max_duration') and args.IsSpecified('graceful_shutdown_max_duration'):
            if graceful_shutdown is None:
                graceful_shutdown = {'maxDuration': args.graceful_shutdown_max_duration}
            else:
                graceful_shutdown['maxDuration'] = args.graceful_shutdown_max_duration
    return graceful_shutdown