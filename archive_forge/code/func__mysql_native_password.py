from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
def _mysql_native_password(cleartext_password):
    mysql_native_encrypted_password = '*' + sha1(sha1(to_bytes(cleartext_password)).digest()).hexdigest().upper()
    return mysql_native_encrypted_password