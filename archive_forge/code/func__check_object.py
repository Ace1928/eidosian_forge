from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _check_object(self, status, model, data, path):
    if data is None:
        return
    if not isinstance(data, dict):
        self._add_invalid_type_report(status, path, '', PropType.OBJECT, data)
        return None
    if PropName.REQUIRED in model:
        self._check_required_fields(status, model[PropName.REQUIRED], data, path)
    model_properties = model[PropName.PROPERTIES]
    for prop in model_properties.keys():
        if prop in data:
            model_prop_val = model_properties[prop]
            expected_type = model_prop_val[PropName.TYPE]
            actually_value = data[prop]
            self._check_types(status, actually_value, expected_type, model_prop_val, path, prop)