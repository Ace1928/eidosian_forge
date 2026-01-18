from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
class ConsulACLNotFoundException(Exception):
    """
    Exception raised if an ACL with is not found.
    """