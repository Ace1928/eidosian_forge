from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import stream_objects
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastream import resource_args
from googlecloudsdk.core import properties
class _StreamObjectInfo:
    """Container for stream object data using in list display."""

    def __init__(self, message, source_object):
        self.display_name = message.displayName
        self.name = message.name
        self.source_object = source_object
        self.backfill_job_state = message.backfillJob.state if message.backfillJob is not None else None
        self.backfill_job_trigger = message.backfillJob.trigger if message.backfillJob is not None else None
        self.last_backfill_job_start_time = message.backfillJob.lastStartTime if message.backfillJob is not None else None
        self.last_backfill_job_end_time = message.backfillJob.lastEndTime if message.backfillJob is not None else None