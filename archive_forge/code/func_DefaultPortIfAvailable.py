from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
def DefaultPortIfAvailable():
    """Returns default port if available.

  Raises:
    EmulatorArgumentsError: if port is not available.

  Returns:
    int, default port
  """
    if portpicker.is_port_free(_DEFAULT_PORT):
        return _DEFAULT_PORT
    else:
        raise EmulatorArgumentsError('Default emulator port [{}] is already in use'.format(_DEFAULT_PORT))