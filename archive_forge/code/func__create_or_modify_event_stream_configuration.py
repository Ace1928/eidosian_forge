from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_event_stream_configuration(job, args, messages):
    """Creates or modifies event stream config. Returns if flag present."""
    event_stream_name = getattr(args, 'event_stream_name', None)
    event_stream_start = getattr(args, 'event_stream_starts', None)
    event_stream_expire = getattr(args, 'event_stream_expires', None)
    if not (event_stream_name or event_stream_start or event_stream_expire):
        return False
    if not job.eventStream:
        job.eventStream = messages.EventStream()
    job.eventStream.name = event_stream_name
    job.eventStream.eventStreamStartTime = event_stream_start
    job.eventStream.eventStreamExpirationTime = event_stream_expire
    return True