from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_name_byschema_ref(self, schema_ref):
    model_name = _get_model_name_from_url(schema_ref)
    model_def = self._definitions[model_name]
    if PropName.ALL_OF in model_def:
        return self._get_model_name_byschema_ref(model_def[PropName.ALL_OF][0][PropName.REF])
    else:
        return model_name