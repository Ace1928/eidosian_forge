from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def IntegrationDeleteStages(destroy_resource_types, should_configure_service):
    """Returns the progress tracker Stages for deleting an Integration.

  Args:
    destroy_resource_types: the set of resource type strings to destroy.
    should_configure_service: bool, Whether a step to configure service binding
      is required.

  Returns:
    list of progress_tracker.Stage
  """
    stages = {}
    if should_configure_service:
        stages[UPDATE_APPLICATION] = progress_tracker.Stage('Unbinding services...', key=UPDATE_APPLICATION)
        stages[CREATE_DEPLOYMENT] = progress_tracker.Stage('Configuring resources...', key=CREATE_DEPLOYMENT)
        service_stages = _DeployStages({'service'}, 'Configuring ')
        stages.update(service_stages)
    stages[UNDEPLOY_RESOURCE] = progress_tracker.Stage('Deleting resources...', key=UNDEPLOY_RESOURCE)
    undeploy_stages = _DeployStages(destroy_resource_types, 'Deleting ')
    stages.update(undeploy_stages)
    stages[CLEANUP_CONFIGURATION] = progress_tracker.Stage('Saving Integration configurations...', key=CLEANUP_CONFIGURATION)
    return stages