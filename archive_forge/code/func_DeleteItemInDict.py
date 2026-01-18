from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def DeleteItemInDict(item, item_path, item_sep='.'):
    """Finds and deletes (potentially) nested value based on specified node_path.

  Args:
      item: Dict, Map like object to search.
      item_path: str, An item_sep separated path to nested item in map.
      item_sep: str, Path item separator, default is '.'.

  Raises:
    KeyError: If item_path not found or empty.
  """
    if not item_path:
        raise KeyError('Missing Path')
    parts = item_path.split(item_sep)
    parts.reverse()
    context = item
    while parts:
        part = parts.pop()
        if part in context and yaml.dict_like(context):
            elem = context.get(part)
            if not parts:
                if elem:
                    del context[part]
                else:
                    raise KeyError('Path [{}] not found'.format(item_path))
            else:
                context = elem
        else:
            raise KeyError('Path [{}] not found'.format(item_path))