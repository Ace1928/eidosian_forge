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
def WriteEnvYaml(env, output_dir):
    """Writes the given environment values into the output_dir/env.yaml file.

  Args:
    env: {str: str}, Dictonary of environment values.
    output_dir: str, Path of directory to which env.yaml file should be written.
  """
    env_file_path = os.path.join(output_dir, 'env.yaml')
    with files.FileWriter(env_file_path) as env_file:
        resource_printer.Print([env], print_format='yaml', out=env_file)