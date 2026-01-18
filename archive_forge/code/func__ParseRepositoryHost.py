from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def _ParseRepositoryHost(repository_name):
    """Parses a repository to an optional hostname and a list of path compoentes.

  Args:
    repository_name: (str) A name made up of slash-separated path name
      components, optionally prefixed by a registry hostname.

  Returns:
    A (hostname, components) tuple representing the parsed result.
    The hostname will be None if it isn't present; the components is a list of
    each slash-separated part in the given repository name.
  """
    components = repository_name.split('/')
    if len(components) == 1:
        return (None, components)
    if '.' in components[0] or ':' in components[0]:
        return (components[0], components[1:])
    return (None, components)