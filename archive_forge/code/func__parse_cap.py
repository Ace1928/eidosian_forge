from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _parse_cap(self, cap, op_required=True):
    opind = -1
    try:
        i = 0
        while opind == -1:
            opind = cap.find(OPS[i])
            i += 1
    except Exception:
        if op_required:
            self.module.fail_json(msg="Couldn't find operator (one of: %s)" % str(OPS))
        else:
            return (cap, None, None)
    op = cap[opind]
    cap, flags = cap.split(op)
    return (cap, op, flags)