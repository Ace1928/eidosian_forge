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
def _create_netbox_object(self, nb_endpoint, data):
    """Create a NetBox object.
        :returns tuple(serialized_nb_obj, diff): tuple of the serialized created
        NetBox object and the Ansible diff.
        """
    if self.check_mode:
        nb_obj = data
    else:
        try:
            nb_obj = nb_endpoint.create(data)
        except pynetbox.RequestError as e:
            self._handle_errors(msg=e.error)
    diff = self._build_diff(before={'state': 'absent'}, after={'state': 'present'})
    return (nb_obj, diff)