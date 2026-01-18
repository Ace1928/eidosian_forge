from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
def _xorder_dn(self):
    regex = '.+\\{\\d+\\}.+'
    explode_dn = ldap.dn.explode_dn(self.module.params['dn'])
    return re.match(regex, explode_dn[0]) is not None