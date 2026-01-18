from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _find_object_matching_params(self, model_name, params):
    get_list_operation = self._find_get_list_operation(model_name)
    if not get_list_operation:
        return None
    data = params[ParamName.DATA]
    if not params.get(ParamName.FILTERS):
        params[ParamName.FILTERS] = {'name': data['name']}
    obj = None
    filtered_objs = self.get_objects_by_filter(get_list_operation, params)
    for i, obj in enumerate(filtered_objs):
        if i > 0:
            raise FtdConfigurationError(MULTIPLE_DUPLICATES_FOUND_ERROR)
    return obj