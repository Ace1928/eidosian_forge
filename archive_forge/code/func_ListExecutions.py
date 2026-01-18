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
def ListExecutions(self, namespace_ref, label_selector='', limit=None, page_size=100):
    """List all executions for the given job.

    Executions list gets sorted by job name, creation timestamp, and completion
    timestamp.

    Args:
      namespace_ref: Resource, namespace to list executions in
      label_selector: Optional[string], extra label selector to filter
        executions
      limit: Optional[int], max number of executions to list.
      page_size: Optional[int], number of executions to fetch at a time

    Yields:
      Executions for the given surface
    """
    messages = self.messages_module
    encoding.AddCustomJsonFieldMapping(messages.RunNamespacesExecutionsListRequest, 'continue_', 'continue')
    request = messages.RunNamespacesExecutionsListRequest(parent=namespace_ref.RelativeName())
    if label_selector:
        request.labelSelector = label_selector
    try:
        for result in list_pager.YieldFromList(service=self._client.namespaces_executions, request=request, limit=limit, batch_size=page_size, current_token_attribute='continue_', next_token_attribute=('metadata', 'continue_'), batch_size_attribute='limit'):
            yield execution.Execution(result, messages)
    except api_exceptions.InvalidDataFromServerError as e:
        serverless_exceptions.MaybeRaiseCustomFieldMismatch(e)