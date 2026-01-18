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
def _GenerateCopyCommand(from_path, to_path, comment=None):
    """Returns a Dockerfile entry that copies a file from host to container.

  Args:
    from_path: (str) Path of the source in host.
    to_path: (str) Path to the destination in the container.
    comment: (str) A comment explaining the copy operation.
  """
    cmd = 'COPY {}\n'.format(json.dumps([from_path, to_path]))
    if comment is not None:
        formatted_comment = '\n# '.join(comment.split('\n'))
        return '# {}\n{}'.format(formatted_comment, cmd)
    return cmd