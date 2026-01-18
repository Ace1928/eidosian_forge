from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def _get_monitor(module):
    if module.params['id'] is not None:
        monitor = api.Monitor.get(module.params['id'])
        if 'errors' in monitor:
            module.fail_json(msg='Failed to retrieve monitor with id %s, errors are %s' % (module.params['id'], str(monitor['errors'])))
        return monitor
    else:
        monitors = api.Monitor.get_all()
        for monitor in monitors:
            if monitor['name'] == _fix_template_vars(module.params['name']):
                return monitor
    return {}