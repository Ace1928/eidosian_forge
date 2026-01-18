from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def _is_bundle_installed(self, bundle):
    try:
        os.stat('/usr/share/clear/bundles/%s' % bundle)
    except OSError:
        return False
    return True