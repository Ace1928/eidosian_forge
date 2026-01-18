from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rundeck import (
class RundeckProjectManager(object):

    def __init__(self, module):
        self.module = module

    def get_project_facts(self):
        resp, info = api_request(module=self.module, endpoint='project/%s' % self.module.params['name'])
        return resp

    def create_or_update_project(self):
        facts = self.get_project_facts()
        if facts is None:
            if self.module.check_mode:
                self.module.exit_json(changed=True, before={}, after={'name': self.module.params['name']})
            resp, info = api_request(module=self.module, endpoint='projects', method='POST', data={'name': self.module.params['name'], 'config': {}})
            if info['status'] == 201:
                self.module.exit_json(changed=True, before={}, after=self.get_project_facts())
            else:
                self.module.fail_json(msg='Unhandled HTTP status %d, please report the bug' % info['status'], before={}, after=self.get_project_facts())
        else:
            self.module.exit_json(changed=False, before=facts, after=facts)

    def remove_project(self):
        facts = self.get_project_facts()
        if facts is None:
            self.module.exit_json(changed=False, before={}, after={})
        else:
            if not self.module.check_mode:
                api_request(module=self.module, endpoint='project/%s' % self.module.params['name'], method='DELETE')
            self.module.exit_json(changed=True, before=facts, after={})