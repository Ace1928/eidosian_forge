from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _add_upserted_object(self, model_operations, params):
    add_op_name = self._get_operation_name(self._operation_checker.is_add_operation, model_operations)
    if not add_op_name:
        raise FtdConfigurationError(ADD_OPERATION_NOT_SUPPORTED_ERROR)
    return self.add_object(add_op_name, params)