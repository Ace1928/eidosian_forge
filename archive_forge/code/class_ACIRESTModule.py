from __future__ import absolute_import, division, print_function
import json
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.aci.plugins.module_utils.aci import ACIModule, aci_argument_spec
from ansible.module_utils._text import to_text
class ACIRESTModule(ACIModule):

    def changed(self, d):
        """Check ACI response for changes"""
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'status' and v in ('created', 'modified', 'deleted'):
                    return True
                elif self.changed(v) is True:
                    return True
        elif isinstance(d, list):
            for i in d:
                if self.changed(i) is True:
                    return True
        return False

    def response_type(self, rawoutput, rest_type='xml'):
        """Handle APIC response output"""
        if rest_type == 'json':
            self.response_json(rawoutput)
        else:
            self.response_xml(rawoutput)
        if HAS_URLPARSE:
            self.result['changed'] = self.changed(self.imdata)