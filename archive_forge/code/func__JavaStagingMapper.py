from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import os
import re
import shutil
import tempfile
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.command_lib.app import jarfile
from googlecloudsdk.command_lib.util import java
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _JavaStagingMapper(command_path, descriptor, app_dir, staging_dir):
    """Map a java staging request to the right args.

  Args:
    command_path: str, path to the jar tool file.
    descriptor: str, path to the `appengine-web.xml`
    app_dir: str, path to the unstaged app directory
    staging_dir: str, path to the empty staging dir

  Raises:
    java.JavaError, if Java is not installed.

  Returns:
    [str], args for executable invocation.
  """
    del descriptor
    java_bin = java.RequireJavaInstalled('local staging for java')
    args = [java_bin, '-classpath', command_path, _JAVA_APPCFG_ENTRY_POINT] + _JAVA_APPCFG_STAGE_FLAGS + ['stage', app_dir, staging_dir]
    return args