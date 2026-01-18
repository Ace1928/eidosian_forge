from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def ValidateTPUState(state, state_enum, tpu_name):
    """Validates the TPU's state.

  Prints warnings or throws exceptions when appropriate.

  Args:
    state: the state of the TPU.
    state_enum: the enum for all TPU states.
    tpu_name: the name of the TPU VM.
  """
    if state is state_enum.READY:
        pass
    elif state is state_enum.STATE_UNSPECIFIED:
        log.warning('The TPU VM "{}" is in state "{}", therefore the TPU may not be available or reachable.'.format(tpu_name, state))
    elif state in [state_enum.CREATING, state_enum.STARTING, state_enum.REPAIRING, state_enum.HIDING, state_enum.UNHIDING]:
        log.warning('The TPU VM "{}" is in state "{}", therefore the TPU may not be available or reachable. If the connection fails, please try again later.'.format(tpu_name, state))
    elif state in [state_enum.STOPPED, state_enum.STOPPING, state_enum.DELETING, state_enum.HIDDEN]:
        raise tpu_exceptions.TPUInUnusableState(state)
    elif state in [state_enum.PREEMPTED, state_enum.TERMINATED]:
        raise tpu_exceptions.TPUInUnusableTerminalState(state)