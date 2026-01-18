from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def activate_ports(module, port_path, ports, stdout, stderr):
    """ Activate a port if it's inactive. """
    activate_c = 0
    for port in ports:
        if not query_port(module, port_path, port):
            module.fail_json(msg='Failed to activate %s, port(s) not present' % port, stdout=stdout, stderr=stderr)
        if query_port(module, port_path, port, state='active'):
            continue
        rc, out, err = module.run_command('%s activate %s' % (port_path, port))
        stdout += out
        stderr += err
        if not query_port(module, port_path, port, state='active'):
            module.fail_json(msg='Failed to activate %s: %s' % (port, err), stdout=stdout, stderr=stderr)
        activate_c += 1
    if activate_c > 0:
        module.exit_json(changed=True, msg='Activated %s port(s)' % activate_c, stdout=stdout, stderr=stderr)
    module.exit_json(changed=False, msg='Port(s) already active', stdout=stdout, stderr=stderr)