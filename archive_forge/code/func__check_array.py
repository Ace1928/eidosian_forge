from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _check_array(self, status, model, data, path):
    if data is None:
        return
    elif not isinstance(data, list):
        self._add_invalid_type_report(status, path, '', PropType.ARRAY, data)
    else:
        item_model = model[PropName.ITEMS]
        for i, item_data in enumerate(data):
            self._check_types(status, item_data, item_model[PropName.TYPE], item_model, '{0}[{1}]'.format(path, i), '')