from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
def _get_gzip_settings_from_command_args(args):
    """Creates GzipSettings object from user flags."""
    if getattr(args, 'gzip_in_flight_all', None):
        return GzipSettings(GzipType.IN_FLIGHT, GZIP_ALL)
    elif getattr(args, 'gzip_in_flight', None):
        return GzipSettings(GzipType.IN_FLIGHT, args.gzip_in_flight)
    elif getattr(args, 'gzip_local_all', None):
        return GzipSettings(GzipType.LOCAL, GZIP_ALL)
    elif getattr(args, 'gzip_local', None):
        return GzipSettings(GzipType.LOCAL, args.gzip_local)
    return None