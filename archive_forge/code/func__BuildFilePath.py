from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _BuildFilePath(output_root, sdk_path, home_directory, *argv):
    path_args = (output_root,) + sdk_path + tuple(home_directory.split('.')) + tuple((path_component for path_component in argv))
    file_path = os.path.join(*path_args)
    return file_path