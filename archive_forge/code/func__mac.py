from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
def _mac(mac):
    """Test if something appears to be a valid mac address"""
    params = {'mac': mac}
    _validate_args('mac', DOCUMENTATION, params)
    re1 = '^([0-9a-f]{2}[:-]){5}[0-9a-f]{2}$'
    re2 = '^([0-9a-f]{4}\\.[0-9a-f]{4}\\.[0-9a-f]{4})$'
    re3 = '^[0-9a-f]{12}$'
    regex = '(?i){re1}|{re2}|{re3}'.format(re1=re1, re2=re2, re3=re3)
    return bool(re.match(regex, mac))