from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import progress_tracker
def _CreateRepoStage():
    return progress_tracker.Stage('Creating Container Repository...', key=CREATE_REPO)