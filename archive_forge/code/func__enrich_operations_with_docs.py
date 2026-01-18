from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _enrich_operations_with_docs(self, operations, docs):

    def get_operation_docs(op):
        op_url = op[OperationField.URL][len(self._base_path):]
        return docs[PropName.PATHS].get(op_url, {}).get(op[OperationField.METHOD], {})
    for operation in operations.values():
        operation_docs = get_operation_docs(operation)
        operation[OperationField.DESCRIPTION] = operation_docs.get(PropName.DESCRIPTION, '')
        if OperationField.PARAMETERS in operation:
            param_descriptions = dict(((p[PropName.NAME], p[PropName.DESCRIPTION]) for p in operation_docs.get(OperationField.PARAMETERS, {})))
            for param_name, params_spec in operation[OperationField.PARAMETERS][OperationParams.PATH].items():
                params_spec[OperationField.DESCRIPTION] = param_descriptions.get(param_name, '')
            for param_name, params_spec in operation[OperationField.PARAMETERS][OperationParams.QUERY].items():
                params_spec[OperationField.DESCRIPTION] = param_descriptions.get(param_name, '')
    return operations