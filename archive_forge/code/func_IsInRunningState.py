from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def IsInRunningState(environment, release_track=base.ReleaseTrack.GA):
    """Returns whether an environment currently is in the RUNNING state.

  Args:
    environment: Environment, an object returned by an API call representing the
      environment to check.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.
  """
    running_state = api_util.GetMessagesModule(release_track=release_track).Environment.StateValueValuesEnum.RUNNING
    return environment.state == running_state