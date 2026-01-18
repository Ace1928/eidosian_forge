from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.util import glob
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _GetIgnoreFileContents(default_ignore_file, directory, include_gitignore=True):
    ignore_file_contents = default_ignore_file
    if include_gitignore and os.path.exists(os.path.join(directory, '.gitignore')):
        ignore_file_contents += '#!include:.gitignore\n'
    return ignore_file_contents