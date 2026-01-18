from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _add_invalid_type_report(self, status, path, prop_name, expected_type, actually_value):
    status[PropName.INVALID_TYPE].append({'path': self._create_path_to_field(path, prop_name), 'expected_type': expected_type, 'actually_value': actually_value})