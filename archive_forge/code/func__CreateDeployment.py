from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def _CreateDeployment(self, appname, tracker, tracker_update_func=None, create_selector=None, delete_selector=None):
    """Create a deployment, waits for operation to finish.

    Args:
      appname:  name of the application.
      tracker: The ProgressTracker to track the deployment operation.
      tracker_update_func: optional custom fn to update the tracker.
      create_selector: create selector for the deployment.
      delete_selector: delete selector for the deployment.
    """
    app_ref = self.GetAppRef(appname)
    deployment_name = self._GetDeploymentName(app_ref.Name())
    if create_selector and delete_selector:
        raise exceptions.ArgumentError('create_selector and delete_selector cannot be specified at the same time.')
    deployment = self.messages.Deployment(name=deployment_name, createSelector=create_selector, deleteSelector=delete_selector, serviceAccount=self._service_account)
    deployment_ops = api_utils.CreateDeployment(self._client, app_ref, deployment)
    dep_response = api_utils.WaitForDeploymentOperation(self._client, deployment_ops, tracker, tracker_update_func=tracker_update_func)
    self.CheckDeploymentState(dep_response)