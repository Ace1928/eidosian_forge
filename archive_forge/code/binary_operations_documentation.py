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
Execute binary and return operation result.

     Will parse args from kwargs into a list of args to pass to underlying
     binary and then attempt to execute it. Will use configured stdout, stderr
     and failure handlers for this operation if configured or module defaults.

    Args:
      cmd: [str], command to be executed with args
      stdin: str, data to send to binary on stdin
      env: {str, str}, environment vars to send to binary.
      **kwargs: mapping of additional arguments to pass to the underlying
        executor.

    Returns:
      OperationResult: execution result for this invocation of the binary.

    Raises:
      ArgumentError, if there is an error parsing the supplied arguments.
      BinaryOperationError, if there is an error executing the binary.
    