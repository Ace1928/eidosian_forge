from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def _CanonicalizePath(secret_path):
    """Canonicalizes secret path to the form `/mount_path:/secret_file_path`.

  Gcloud secret path is more restrictive than the backend (shortn/_bwgb3xdRxL).
  Paths are reduced to their canonical forms before the request is made.

  Args:
    secret_path: Complete path to the secret.

  Returns:
    Canonicalized secret path.
  """
    secret_path = re.sub('/+', '/', secret_path)
    seperator = ':' if ':' in secret_path else '/'
    mount_path, _, secret_file_path = secret_path.rpartition(seperator)
    mount_path = re.sub('/$', '', mount_path)
    secret_file_path = re.sub('^/', '', secret_file_path)
    return '{}:/{}'.format(mount_path, secret_file_path)