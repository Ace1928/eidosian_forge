from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import socket
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.runtime_config import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.runtime_config import flags
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
def _RaiseTimeout():
    raise exceptions.OperationTimeoutError('Variable did not change prior to timeout.', exit_code=2)