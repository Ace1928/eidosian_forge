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
def _delete_netbox_object(self):
    """Delete a NetBox object.
        :returns diff (dict): Ansible diff
        """
    if not self.check_mode:
        try:
            self.nb_object.delete()
        except pynetbox.RequestError as e:
            self._handle_errors(msg=e.error)
    diff = self._build_diff(before={'state': 'present'}, after={'state': 'absent'})
    return diff