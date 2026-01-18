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
def ReadEnvYaml(output_dir):
    """Reads and returns the environment values in output_dir/env.yaml file.

  Args:
    output_dir: str, Path of directory containing the env.yaml to be read.

  Returns:
    env: {str: str}
  """
    env_file_path = os.path.join(output_dir, 'env.yaml')
    try:
        with files.FileReader(env_file_path) as f:
            return yaml.load(f)
    except files.MissingFileError:
        raise NoEnvYamlError(output_dir)