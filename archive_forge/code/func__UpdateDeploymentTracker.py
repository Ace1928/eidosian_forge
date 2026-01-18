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
@staticmethod
def _UpdateDeploymentTracker(tracker, operation, tracker_stages):
    """Updates deployment tracker with the current status of operation.

    Args:
      tracker: The ProgressTracker to track the deployment operation.
      operation: run_apps.v1alpha1.operation object for the deployment.
      tracker_stages: map of stages with key as stage key (string) and value is
        the progress_tracker.Stage.
    """
    messages = api_utils.GetMessages()
    metadata = api_utils.GetDeploymentOperationMetadata(messages, operation)
    resources_in_progress = []
    resources_completed = []
    resource_state = messages.ResourceDeploymentStatus.StateValueValuesEnum
    if metadata.resourceStatus is not None:
        for resource in metadata.resourceStatus:
            stage_name = stages.StageKeyForResourceDeployment(resource.name.type)
            if resource.state == resource_state.RUNNING:
                resources_in_progress.append(stage_name)
            if resource.state == resource_state.FINISHED:
                resources_completed.append(stage_name)
    for resource in resources_in_progress:
        if resource in tracker_stages and tracker.IsWaiting(resource):
            tracker.StartStage(resource)
    for resource in resources_completed:
        if resource in tracker_stages and tracker.IsRunning(resource):
            tracker.CompleteStage(resource)
    tracker.Tick()