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
def _SecretsDiffer(project1, secret1, project2, secret2):
    """Returns true if the two secrets differ.

  The secrets can be considered as different if either the secret name is
  different or the project is different with the secret name being the same. If
  one project is represented using the project number and the other is
  represented using its project id, then it may not be possible to determine if
  the two projects are the same, so the validation is relaxed.

  Args:
    project1: Project ID or number of the first secret.
    secret1: Secret name of the first secret.
    project2: Project ID or number of the second secret.
    secret2: Secret name of the second secret.

  Returns:
    True if the two secrets differ, False otherwise.
  """
    return secret1 != secret2 or (project1 != project2 and project1.isdigit() == project2.isdigit() and (project1 != _DEFAULT_PROJECT_IDENTIFIER) and (project2 != _DEFAULT_PROJECT_IDENTIFIER))