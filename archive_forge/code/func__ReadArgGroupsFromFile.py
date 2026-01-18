from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def _ReadArgGroupsFromFile(arg_file):
    """Collects all the arg groups defined in the yaml file into a dictionary.

  Each dictionary key is an arg-group name whose corresponding value is a nested
  dictionary containing arg-name: arg-value pairs defined in that group.

  Args:
    arg_file: str, the name of the YAML argument file to open and parse.

  Returns:
    A dict containing all arg-groups found in the arg_file.

  Raises:
    yaml.Error: If the YAML file could not be read or parsed.
    BadFileException: If the contents of the file are not valid.
  """
    all_groups = {}
    for d in yaml.load_all_path(arg_file):
        if d is None:
            log.warning('Ignoring empty yaml document.')
        elif isinstance(d, dict):
            all_groups.update(d)
        else:
            raise calliope_exceptions.BadFileException('Failed to parse YAML file [{}]: [{}] is not a valid argument group.'.format(arg_file, six.text_type(d)))
    return all_groups