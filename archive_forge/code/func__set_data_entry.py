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
def _set_data_entry(self, instance_name, key, value, path=None):
    """Helper to save data

        Helper to save the data in self.data
        Detect if data is already in branch and use dict_merge() to prevent that branch is overwritten.

        Args:
            str(instance_name): name of instance
            str(key): same as dict
            *(value): same as dict
        Kwargs:
            str(path): path to branch-part
        Raises:
            AnsibleParserError
        Returns:
            None"""
    if not path:
        path = self.data['inventory']
    if instance_name not in path:
        path[instance_name] = {}
    try:
        if isinstance(value, dict) and key in path[instance_name]:
            path[instance_name] = dict_merge(value, path[instance_name][key])
        else:
            path[instance_name][key] = value
    except KeyError as err:
        raise AnsibleParserError('Unable to store Information: {0}'.format(to_native(err)))