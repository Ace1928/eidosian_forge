from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def _TransformRequestedRunDuration(resource):
    """Properly format requested_run_duration field."""
    run_duration = resource.get('requestedRunDuration', {})
    if not run_duration:
        return ''
    seconds = int(run_duration.get('seconds'))
    days = seconds // 86400
    seconds -= days * 86400
    hours = seconds // 3600
    seconds -= hours * 3600
    minutes = seconds // 60
    seconds -= minutes * 60
    duration = iso_duration.Duration(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return times.FormatDuration(duration, parts=-1)