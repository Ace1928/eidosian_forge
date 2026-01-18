from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import dict_merge
def _extract_schema_from_doc(self, doc_obj, temp_schema):
    """Extract the schema from a doc string
        :param doc_obj: The doc as a python obj
        :type doc_obj: dictionary
        :params temp_schema: The dict in which we stuff the schema parts
        :type temp_schema: dict
        """
    options_obj = doc_obj.get('options')
    for okey, ovalue in iteritems(options_obj):
        temp_schema[okey] = {}
        for metakey in list(ovalue):
            if metakey == 'suboptions':
                temp_schema[okey].update({'options': {}})
                suboptions_obj = {'options': ovalue['suboptions']}
                self._extract_schema_from_doc(suboptions_obj, temp_schema[okey]['options'])
            elif metakey in OPTION_METADATA + OPTION_CONDITIONALS:
                temp_schema[okey].update({metakey: ovalue[metakey]})