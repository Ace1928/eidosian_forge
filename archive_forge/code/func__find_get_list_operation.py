from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _find_get_list_operation(self, model_name):
    operations = self.get_operation_specs_by_model_name(model_name) or {}
    return next((op for op, op_spec in operations.items() if self._operation_checker.is_get_list_operation(op, op_spec)), None)