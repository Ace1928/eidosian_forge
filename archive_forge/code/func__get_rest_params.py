from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_rest_params(self, params):
    path = {}
    query = {}
    operation_param = {OperationParams.PATH: path, OperationParams.QUERY: query}
    for param in params:
        in_param = param['in']
        if in_param == OperationParams.QUERY:
            query[param[PropName.NAME]] = self._simplify_param_def(param)
        elif in_param == OperationParams.PATH:
            path[param[PropName.NAME]] = self._simplify_param_def(param)
    return operation_param