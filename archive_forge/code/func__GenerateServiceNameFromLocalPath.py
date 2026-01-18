from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import os
import re
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util as concepts_util
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _GenerateServiceNameFromLocalPath(source):
    """Produce a valid default service name from a local file or directory path.

  Converts a file or directory path into a reasonable default service name by
  resolving relative paths to absolute paths, removing any extensions, and then
  removing any invalid characters.

  For example, the paths /tmp/foo/bar/.. and /tmp/foo.tar.gz would both produce
  the service name 'foo'. A source path of "." will be expanded to the current
  directory name."

  Args:
    source: str, The file or directory path.

  Returns:
    A valid Cloud Run service name.
  """
    path, ext = os.path.splitext(os.path.abspath(source))
    while ext:
        path, ext = os.path.splitext(path)
    return _GenerateServiceName(path)