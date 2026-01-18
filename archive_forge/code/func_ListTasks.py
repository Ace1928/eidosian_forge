from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def ListTasks(self, namespace_ref, execution_name, include_states=None, limit=None, page_size=100):
    """List all tasks for the given execution.

    Args:
      namespace_ref: Resource, namespace to list tasks in
      execution_name: str, The execution for which to list tasks.
      include_states: List[str], states of tasks to include in the list.
      limit: Optional[int], max number of tasks to list.
      page_size: Optional[int], number of tasks to fetch at a time

    Yields:
      Executions for the given surface
    """
    messages = self.messages_module
    encoding.AddCustomJsonFieldMapping(messages.RunNamespacesTasksListRequest, 'continue_', 'continue')
    request = messages.RunNamespacesTasksListRequest(parent=namespace_ref.RelativeName())
    label_selectors = []
    if execution_name is not None:
        label_selectors.append('{label} = {name}'.format(label=task.EXECUTION_LABEL, name=execution_name))
    if include_states is not None:
        status_selector = '{label} in ({states})'.format(label=task.STATE_LABEL, states=','.join(include_states))
        label_selectors.append(status_selector)
    if label_selectors:
        request.labelSelector = ','.join(label_selectors)
    try:
        for result in list_pager.YieldFromList(service=self._client.namespaces_tasks, request=request, limit=limit, batch_size=page_size, current_token_attribute='continue_', next_token_attribute=('metadata', 'continue_'), batch_size_attribute='limit'):
            yield task.Task(result, messages)
    except api_exceptions.InvalidDataFromServerError as e:
        serverless_exceptions.MaybeRaiseCustomFieldMismatch(e)