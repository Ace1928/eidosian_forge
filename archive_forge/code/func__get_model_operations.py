from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_operations(self, operations):
    model_operations = {}
    for operations_name, params in iteritems(operations):
        model_name = params[OperationField.MODEL_NAME]
        model_operations.setdefault(model_name, {})[operations_name] = params
    return model_operations