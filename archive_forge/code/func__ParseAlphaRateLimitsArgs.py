from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def _ParseAlphaRateLimitsArgs(args, queue_type, messages, is_update):
    """Parses the attributes of 'args' for Queue.rateLimits."""
    if queue_type == constants.PUSH_QUEUE and _AnyArgsSpecified(args, ['max_tasks_dispatched_per_second', 'max_concurrent_tasks'], clear_args=is_update):
        return messages.RateLimits(maxTasksDispatchedPerSecond=args.max_tasks_dispatched_per_second, maxConcurrentTasks=args.max_concurrent_tasks)