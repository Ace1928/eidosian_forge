from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.six import raise_from
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _get_data_entry(self, path, data=None, delimiter='/'):
    """Helper to get data

        Helper to get data from self.data by a path like 'path/to/target'
        Attention: Escaping of the delimiter is not (yet) provided.

        Args:
            str(path): path to nested dict
        Kwargs:
            dict(data): datastore
            str(delimiter): delimiter in Path.
        Raises:
            None
        Returns:
            *(value)"""
    try:
        if not data:
            data = self.data
        if delimiter in path:
            path = path.split(delimiter)
        if isinstance(path, list) and len(path) > 1:
            data = data[path.pop(0)]
            path = delimiter.join(path)
            return self._get_data_entry(path, data, delimiter)
        return data[path]
    except KeyError:
        return None