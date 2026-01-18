from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _check_enum(self, status, model, value, path):
    if value is not None and value not in model[PropName.ENUM]:
        self._add_invalid_type_report(status, path, '', PropName.ENUM, value)