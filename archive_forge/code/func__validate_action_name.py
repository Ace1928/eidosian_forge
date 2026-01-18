from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
def _validate_action_name(self):
    if self.type in [WORKFLOW_RUNNABLE_TYPE, PLUGIN_RUNNABLE_TYPE]:
        res = self.client.runnable.get_available_actions(self.type, self.name)
        available_actions = res[rest_client.RESP_DATA]['names']
        if self.action_name not in available_actions:
            raise WrongActionNameException(self.action_name, available_actions)