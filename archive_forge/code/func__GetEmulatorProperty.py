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
def _GetEmulatorProperty(prefix, prop_name):
    """Returns the value of the given property in the given emulator group.

  Args:
    prefix: str, The prefix for the *_emulator property group to look up.
    prop_name: str, The name of the property to look up.

  Returns:
    str, The the value of the given property in the specified emulator group.
  """
    property_group = 'emulator'
    full_name = '{}_{}'.format(prefix, prop_name)
    for section in properties.VALUES:
        if section.name == property_group and section.HasProperty(full_name):
            return section.Property(full_name).Get()
    return None