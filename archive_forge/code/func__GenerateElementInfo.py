from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
def _GenerateElementInfo(impl_path, names):
    """Generates the data a group needs to load sub elements.

  Args:
    impl_path: The file path to the command implementation for this group.
    names: [str], The names of the sub groups or commands found in the group.

  Raises:
    LayoutException: if there is a command or group with an illegal name.

  Returns:
    {str: [str], A mapping from name to a list of paths that implement that
    command or group. There can be multiple paths because a command or group
    could be implemented in both python and yaml (for different release tracks).
  """
    elements = {}
    for name in names:
        if re.search('[A-Z]', name):
            raise LayoutException('Commands and groups cannot have capital letters: {0}.'.format(name))
        cli_name = name[:-5] if name.endswith('.yaml') else name
        sub_path = os.path.join(impl_path, name)
        existing = elements.setdefault(cli_name, [])
        existing.append(sub_path)
    return elements