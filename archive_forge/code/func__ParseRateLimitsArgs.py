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
def _ParseRateLimitsArgs(args, queue_type, messages, is_update):
    """Parses the attributes of 'args' for Queue.rateLimits."""
    if queue_type == constants.PUSH_QUEUE and _AnyArgsSpecified(args, ['max_dispatches_per_second', 'max_concurrent_dispatches', 'max_burst_size'], clear_args=is_update):
        max_burst_size = args.max_burst_size if hasattr(args, 'max_burst_size') else None
        return messages.RateLimits(maxDispatchesPerSecond=args.max_dispatches_per_second, maxConcurrentDispatches=args.max_concurrent_dispatches, maxBurstSize=max_burst_size)