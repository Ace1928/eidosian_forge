from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def _GenerateShortOrLongBackendNames(metastore_type_and_name):
    """Validate and process the format of short and long names for backends.

  Args:
    metastore_type_and_name: Metastore type and name.

  Returns:
    String backend name.

  Raises:
    BadArgumentException: When the input backend(s) are invalid
  """
    if metastore_type_and_name[0].lower() == 'bigquery':
        long_name_regex = '^projects\\/.*[^\\/]'
    else:
        long_name_regex = '^projects\\/.*[^\\/]\\/locations\\/.[^\\/]*\\/(' + _GetMetastoreTypeFromDict(METASTORE_RESOURCE_PATH_DICT) + ')\\/.[^\\/]*$'
    if '/' in metastore_type_and_name[1]:
        if re.search(long_name_regex, metastore_type_and_name[1]):
            return metastore_type_and_name[1]
        else:
            raise exceptions.BadArgumentException('--backends', 'Invalid backends format')
    elif metastore_type_and_name[0].lower() == 'bigquery':
        return 'projects/' + metastore_type_and_name[1]
    else:
        return '{0}/' + METASTORE_RESOURCE_PATH_DICT[metastore_type_and_name[0]] + '/' + metastore_type_and_name[1]