from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_name_from_responses(self, params):
    responses = params[PropName.RESPONSES]
    if SUCCESS_RESPONSE_CODE in responses:
        response = responses[SUCCESS_RESPONSE_CODE][PropName.SCHEMA]
        if PropName.REF in response:
            return self._get_model_name_byschema_ref(response[PropName.REF])
        elif PropName.PROPERTIES in response:
            ref = response[PropName.PROPERTIES][PropName.ITEMS][PropName.ITEMS][PropName.REF]
            return self._get_model_name_byschema_ref(ref)
        elif PropName.TYPE in response and response[PropName.TYPE] == PropType.FILE:
            return FILE_MODEL_NAME
    else:
        return None