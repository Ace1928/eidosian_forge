from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
import re
import textwrap
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import log
from six.moves import shlex_quote
def _GenerateEntrypoint(package, is_prebuilt_image=False):
    """Generates dockerfile entry to set the container entrypoint.

  Args:
    package: (Package) Represents the main application copied to the container.
    is_prebuilt_image: (bool) Whether the base image is pre-built and provided
      by Vertex AI.

  Returns:
    A string with Dockerfile directives to set ENTRYPOINT
  """
    python_command = 'python3' if is_prebuilt_image else 'python'
    if package.python_module is not None:
        exec_str = json.dumps([python_command, '-m', package.python_module])
    else:
        _, ext = os.path.splitext(package.script)
        executable = [python_command] if ext == '.py' else ['/bin/bash']
        exec_str = json.dumps(executable + [package.script])
    return '\nENTRYPOINT {}'.format(exec_str)