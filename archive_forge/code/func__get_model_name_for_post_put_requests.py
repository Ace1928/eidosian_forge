from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_name_for_post_put_requests(self, params):
    model_name = None
    if OperationField.PARAMETERS in params:
        body_param_dict = self._get_body_param_from_parameters(params[OperationField.PARAMETERS])
        if body_param_dict:
            schema_ref = body_param_dict[PropName.SCHEMA][PropName.REF]
            model_name = self._get_model_name_byschema_ref(schema_ref)
    if model_name is None:
        model_name = self._get_model_name_from_responses(params)
    return model_name