from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateBackendsAndReturnMetastoreDict(backends):
    """Validate backends argument if it has correct format, metastore type and the keys are positive number and not duplicated.

  In addition, parsing the backends to backend metastore dict

  Args:
    backends: A string is passed by user in format
      <key>=<metastore_type>:<name>,... For example:
      1=dpms:dpms1,2=dataplex:lake1

  Returns:
    Backend metastore dict
  Raises:
    BadArgumentException: When the input backends is invalid or duplicated keys
  """
    backend_dict = {}
    if not backends:
        raise exceptions.BadArgumentException('--backends', 'Cannot be empty')
    backend = backends.split(',')
    for data in backend:
        rank_and_metastore = data.split('=')
        if len(rank_and_metastore) != 2:
            raise exceptions.BadArgumentException('--backends', 'Invalid backends format')
        key = rank_and_metastore[0]
        if not _IsZeroOrPositiveNumber(key):
            raise exceptions.BadArgumentException('--backends', 'Invalid backends format or key of backend is less than 0')
        value = rank_and_metastore[1]
        metastore_type_and_name = value.split(':')
        if len(metastore_type_and_name) != 2:
            raise exceptions.BadArgumentException('--backends', 'Invalid backends format')
        if key in backend_dict:
            raise exceptions.BadArgumentException('--backends', 'Duplicated keys of backends')
        if metastore_type_and_name[0] not in METASTORE_TYPE_DICT.keys():
            raise exceptions.BadArgumentException('--backends', 'Invalid backends type')
        generated_name = _GenerateShortOrLongBackendNames(metastore_type_and_name)
        backend_metastores_dict = {'name': generated_name, 'metastoreType': METASTORE_TYPE_DICT[metastore_type_and_name[0]]}
        backend_dict[key] = backend_metastores_dict
    return backend_dict