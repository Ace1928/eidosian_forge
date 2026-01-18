import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _format_sort_param(self, sort, resource_type=None):
    """Formats the sort information into the sort query string parameter.

        The input sort information can be any of the following:
        - Comma-separated string in the form of <key[:dir]>
        - List of strings in the form of <key[:dir]>
        - List of either string keys, or tuples of (key, dir)

        For example, the following import sort values are valid:
        - 'key1:dir1,key2,key3:dir3'
        - ['key1:dir1', 'key2', 'key3:dir3']
        - [('key1', 'dir1'), 'key2', ('key3', dir3')]

        :param sort: Input sort information
        :returns: Formatted query string parameter or None
        :raise ValueError: If an invalid sort direction or invalid sort key is
                           given
        """
    if not sort:
        return None
    if isinstance(sort, str):
        sort = [s for s in sort.split(',') if s]
    sort_array = []
    for sort_item in sort:
        sort_key, _sep, sort_dir = sort_item.partition(':')
        sort_key = sort_key.strip()
        sort_key = self._format_sort_key_param(sort_key, resource_type)
        if sort_dir:
            sort_dir = sort_dir.strip()
            if sort_dir not in SORT_DIR_VALUES:
                msg = 'sort_dir must be one of the following: %s.' % ', '.join(SORT_DIR_VALUES)
                raise ValueError(msg)
            sort_array.append('%s:%s' % (sort_key, sort_dir))
        else:
            sort_array.append(sort_key)
    return ','.join(sort_array)