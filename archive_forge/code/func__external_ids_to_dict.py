from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _external_ids_to_dict(text):
    if not text:
        return None
    else:
        d = {}
        for l in text.splitlines():
            if l:
                k, v = l.split('=')
                d[k] = v
        return d