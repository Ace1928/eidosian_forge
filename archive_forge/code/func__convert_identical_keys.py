from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _convert_identical_keys(self, data):
    """
        Used to change non-clashing keys for each module into identical keys that are required
        to be passed to pynetbox
        ex. rack_role back into role to pass to NetBox
        Returns data
        :params data (dict): Data dictionary after _find_ids method ran
        """
    temp_dict = dict()
    if self._version_check_greater(self.version, '2.7', greater_or_equal=True):
        if data.get('form_factor'):
            temp_dict['type'] = data.pop('form_factor')
    for key in data:
        if self.endpoint == 'power_panels' and key == 'rack_group':
            temp_dict[key] = data[key]
        elif key == 'device_role' and (not self._version_check_greater(self.version, '3.6', greater_or_equal=True)):
            temp_dict[key] = data[key]
        elif key in CONVERT_KEYS:
            if key in ('assigned_object', 'scope', 'component'):
                temp_dict[key] = data[key]
            new_key = CONVERT_KEYS[key]
            temp_dict[new_key] = data[key]
        else:
            temp_dict[key] = data[key]
    return temp_dict