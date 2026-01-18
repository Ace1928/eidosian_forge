from __future__ import absolute_import, division, print_function
import crypt
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
def _hash_password(self, password):
    method = crypt.METHOD_SHA512
    salt = crypt.mksalt(method, rounds=10000)
    pw_hash = crypt.crypt(password, salt)
    return pw_hash