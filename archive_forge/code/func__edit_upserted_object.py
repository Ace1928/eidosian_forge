from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _edit_upserted_object(self, model_operations, existing_object, params):
    edit_op_name = self._get_operation_name(self._operation_checker.is_edit_operation, model_operations)
    _set_default(params, 'path_params', {})
    _set_default(params, 'data', {})
    params['path_params']['objId'] = existing_object['id']
    copy_identity_properties(existing_object, params['data'])
    return self.edit_object(edit_op_name, params)