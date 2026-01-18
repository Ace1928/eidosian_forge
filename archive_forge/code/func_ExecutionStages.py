from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import progress_tracker
def ExecutionStages(include_completion=False):
    """Returns the list of progress tracker Stages for Executions."""
    stages = [progress_tracker.Stage('Provisioning resources...', key=_RESOURCES_AVAILABLE)]
    if include_completion:
        stages.append(progress_tracker.Stage('Starting execution...', key=_STARTED))
        stages.append(progress_tracker.Stage('Running execution...', key=_COMPLETED))
    return stages