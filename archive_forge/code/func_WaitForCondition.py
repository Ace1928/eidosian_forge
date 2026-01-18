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
def WaitForCondition(self, poller, max_wait_ms=0):
    """Wait for a configuration to be ready in latest revision.

    Args:
      poller: A ConditionPoller object.
      max_wait_ms: int, if not 0, passed to waiter.PollUntilDone.

    Returns:
      A condition.Conditions object.

    Raises:
      RetryException: Max retry limit exceeded.
      ConfigurationError: configuration failed to
    """
    try:
        if max_wait_ms == 0:
            return waiter.PollUntilDone(poller, None, wait_ceiling_ms=1000)
        return waiter.PollUntilDone(poller, None, max_wait_ms=max_wait_ms, wait_ceiling_ms=1000)
    except retry.RetryException as err:
        conditions = poller.GetConditions()
        msg = conditions.DescriptiveMessage() if conditions else None
        if msg:
            log.error('Still waiting: {}'.format(msg))
        raise err