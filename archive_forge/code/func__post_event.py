from __future__ import absolute_import, division, print_function
import platform
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def _post_event(module):
    try:
        if module.params['host'] is None:
            module.params['host'] = platform.node().split('.')[0]
        msg = api.Event.create(title=module.params['title'], text=module.params['text'], host=module.params['host'], tags=module.params['tags'], priority=module.params['priority'], alert_type=module.params['alert_type'], aggregation_key=module.params['aggregation_key'], source_type_name='ansible')
        if msg['status'] != 'ok':
            module.fail_json(msg=msg)
        module.exit_json(changed=True, msg=msg)
    except Exception as e:
        module.fail_json(msg=to_native(e), exception=traceback.format_exc())