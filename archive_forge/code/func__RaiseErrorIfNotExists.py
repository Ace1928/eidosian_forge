from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.core.util import files
def _RaiseErrorIfNotExists(local_package_path, flag_name):
    """Validate the local package is valid.

  Args:
    local_package_path: str, path of the local directory to check.
    flag_name: str, indicates in which flag the path is specified.
  """
    work_dir = os.path.abspath(files.ExpandHomeDir(local_package_path))
    if not os.path.exists(work_dir) or not os.path.isdir(work_dir):
        raise exceptions.InvalidArgumentException(flag_name, "Directory '{}' is not found.".format(work_dir))