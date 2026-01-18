from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _enrich_definitions_with_docs(self, definitions, docs):
    for model_name, model_def in definitions.items():
        model_docs = docs[SpecProp.DEFINITIONS].get(model_name, {})
        model_def[PropName.DESCRIPTION] = model_docs.get(PropName.DESCRIPTION, '')
        for prop_name, prop_spec in model_def.get(PropName.PROPERTIES, {}).items():
            prop_spec[PropName.DESCRIPTION] = model_docs.get(PropName.PROPERTIES, {}).get(prop_name, '')
            prop_spec[PropName.REQUIRED] = prop_name in model_def.get(PropName.REQUIRED, [])
    return definitions