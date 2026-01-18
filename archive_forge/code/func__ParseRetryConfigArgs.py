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
def _ParseRetryConfigArgs(args, queue_type, messages, is_update, is_alpha=False):
    """Parses the attributes of 'args' for Queue.retryConfig."""
    if queue_type == constants.PULL_QUEUE and _AnyArgsSpecified(args, ['max_attempts', 'max_retry_duration'], clear_args=is_update):
        retry_config = messages.RetryConfig(maxRetryDuration=args.max_retry_duration)
        _AddMaxAttemptsFieldsFromArgs(args, retry_config, is_alpha)
        return retry_config
    if queue_type == constants.PUSH_QUEUE and _AnyArgsSpecified(args, ['max_attempts', 'max_retry_duration', 'max_doublings', 'min_backoff', 'max_backoff'], clear_args=is_update):
        retry_config = messages.RetryConfig(maxRetryDuration=args.max_retry_duration, maxDoublings=args.max_doublings, minBackoff=args.min_backoff, maxBackoff=args.max_backoff)
        _AddMaxAttemptsFieldsFromArgs(args, retry_config, is_alpha)
        return retry_config