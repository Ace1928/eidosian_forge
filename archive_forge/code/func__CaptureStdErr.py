from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _CaptureStdErr(result_holder, output_message=None, raw_output=None):
    """Update OperationResult either from OutputMessage or plain text."""
    if not result_holder.stderr:
        result_holder.stderr = []
    if output_message:
        if output_message.body:
            result_holder.stderr.append(output_message.body)
        if output_message.IsError():
            result_holder.stderr.append(output_message.error_details.Format())
    elif raw_output:
        result_holder.stderr.append(raw_output)