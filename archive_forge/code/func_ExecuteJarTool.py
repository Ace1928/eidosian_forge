from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def ExecuteJarTool(java_bin, jar_dir, jar_name, classname, flags=None, *args):
    """Execute a given jar with the given args and command line.

  Args:
    java_bin: str, path to the system Java binary
    jar_dir: str, the directory the jar is located in
    jar_name: str, file name of the jar under tool_dir
    classname: str, name of the main class in the jar
    flags: [str], flags for the java binary
    *args: args for the command
  """
    flags = flags or []
    jar_path = _FullPath(jar_dir, jar_name)
    classname_arg = [classname] if classname else []
    java_args = ['-cp', jar_path] + flags + classname_arg + list(args)
    _ExecuteTool(execution_utils.ArgsForExecutableTool(java_bin, *java_args))