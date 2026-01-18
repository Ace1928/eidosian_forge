from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import progress_tracker
def JobStages(execute_now=False, include_completion=False, include_build=False, include_create_repo=False):
    """Returns the list of progress tracker Stages for Jobs."""
    stages = []
    if include_create_repo:
        stages.append(_CreateRepoStage())
    if include_build:
        stages.append(_UploadSourceStage())
        stages.append(_BuildContainerStage())
    if execute_now:
        stages += ExecutionStages(include_completion)
    return stages