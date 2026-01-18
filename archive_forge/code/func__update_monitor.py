from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def _update_monitor(module, monitor, options):
    try:
        kwargs = dict(id=monitor['id'], query=module.params['query'], name=_fix_template_vars(module.params['name']), message=_fix_template_vars(module.params['notification_message']), escalation_message=_fix_template_vars(module.params['escalation_message']), priority=module.params['priority'], options=options)
        if module.params['tags'] is not None:
            kwargs['tags'] = module.params['tags']
        msg = api.Monitor.update(**kwargs)
        if 'errors' in msg:
            module.fail_json(msg=str(msg['errors']))
        elif _equal_dicts(msg, monitor, ['creator', 'overall_state', 'modified', 'matching_downtimes', 'overall_state_modified']):
            module.exit_json(changed=False, msg=msg)
        else:
            module.exit_json(changed=True, msg=msg)
    except Exception as e:
        module.fail_json(msg=to_native(e), exception=traceback.format_exc())