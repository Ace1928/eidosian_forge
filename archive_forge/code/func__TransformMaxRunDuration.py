from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core.util.iso_duration import Duration
from googlecloudsdk.core.util.times import FormatDuration
def _TransformMaxRunDuration(resource):
    """Properly format max_run_duration field."""
    bulk_resource = resource.get('bulkInsertInstanceResource')
    max_run_duration = bulk_resource.get('instanceProperties', {}).get('scheduling', {}).get('maxRunDuration')
    if not max_run_duration:
        return ''
    duration = Duration(seconds=int(max_run_duration.get('seconds')))
    return FormatDuration(duration, parts=4)