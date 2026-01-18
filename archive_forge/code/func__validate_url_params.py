from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _validate_url_params(self, operation, params, resource):
    if params is None:
        params = {}
    self._check_validate_url_params(operation, params)
    operation = self._operations[operation]
    if OperationField.PARAMETERS in operation and resource in operation[OperationField.PARAMETERS]:
        spec = operation[OperationField.PARAMETERS][resource]
        status = self._init_report()
        self._check_url_params(status, spec, params)
        if len(status[PropName.REQUIRED]) > 0 or len(status[PropName.INVALID_TYPE]) > 0:
            return (False, self._delete_empty_field_from_report(status))
        return (True, None)
    else:
        return (True, None)