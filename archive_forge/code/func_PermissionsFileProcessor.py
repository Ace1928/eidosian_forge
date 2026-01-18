from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def PermissionsFileProcessor(input_file):
    """Builds a bigquery AccessValueListEntry array from input file.

  Expects YAML or JSON formatted file.

  Args:
    input_file: input file contents from argparse namespace.

  Raises:
    PermissionsFileError: if the file contents are not a valid JSON or YAML
      file.

  Returns:
    [AccessValueListEntry]: Array of AccessValueListEntry messages specifying
      access permissions defined input file.
  """
    access_value_msg = GetApiMessage('Dataset').AccessValueListEntry
    try:
        permissions_array = []
        permissions_from_file = yaml.load(input_file[0])
        permissions_from_file = permissions_from_file.get('access', None)
        if not permissions_from_file or not isinstance(permissions_from_file, list):
            raise PermissionsFileError('Error parsing permissions file: no access list defined in file')
        for access_yaml in permissions_from_file:
            permission = encoding.PyValueToMessage(access_value_msg, access_yaml)
            if _ValidatePermission(permission):
                permissions_array.append(permission)
            else:
                raise PermissionsFileError('Error parsing permissions file: invalid permission definition [{}]'.format(permission))
        return sorted(permissions_array, key=lambda x: x.role)
    except yaml.YAMLParseError as ype:
        raise PermissionsFileError('Error parsing permissions file [{}]'.format(ype))