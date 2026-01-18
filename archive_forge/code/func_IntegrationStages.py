from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def IntegrationStages(create, resource_types):
    """Returns the progress tracker Stages for creating or updating an Integration.

  Args:
    create: whether it's for the create command.
    resource_types: set of resource type strings to deploy.

  Returns:
    dict of stage key to progress_tracker Stage.
  """
    stages = {UPDATE_APPLICATION: _UpdateApplicationStage(create)}
    stages[CREATE_DEPLOYMENT] = progress_tracker.Stage('Configuring Integration...', key=CREATE_DEPLOYMENT)
    deploy_stages = _DeployStages(resource_types, 'Configuring ')
    stages.update(deploy_stages)
    return stages