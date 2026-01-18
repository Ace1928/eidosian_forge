from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def _UpdateApplicationStage(create):
    """Returns the stage for updating the Application.

  Args:
    create: whether it's for the create command.

  Returns:
    progress_tracker.Stage
  """
    if create:
        message = 'Saving Configuration for Integration...'
    else:
        message = 'Updating Configuration for Integration...'
    return progress_tracker.Stage(message, key=UPDATE_APPLICATION)