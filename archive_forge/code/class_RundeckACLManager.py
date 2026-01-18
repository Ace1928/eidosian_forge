from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rundeck import (
class RundeckACLManager:

    def __init__(self, module):
        self.module = module

    def get_acl(self):
        resp, info = api_request(module=self.module, endpoint='system/acl/%s.aclpolicy' % self.module.params['name'])
        return resp

    def create_or_update_acl(self):
        facts = self.get_acl()
        if facts is None:
            if self.module.check_mode:
                self.module.exit_json(changed=True, before={}, after=self.module.params['policy'])
            resp, info = api_request(module=self.module, endpoint='system/acl/%s.aclpolicy' % self.module.params['name'], method='POST', data={'contents': self.module.params['policy']})
            if info['status'] == 201:
                self.module.exit_json(changed=True, before={}, after=self.get_acl())
            elif info['status'] == 400:
                self.module.fail_json(msg="Unable to validate acl %s. Please ensure it's a valid ACL" % self.module.params['name'])
            elif info['status'] == 409:
                self.module.fail_json(msg='ACL %s already exists' % self.module.params['name'])
            else:
                self.module.fail_json(msg='Unhandled HTTP status %d, please report the bug' % info['status'], before={}, after=self.get_acl())
        else:
            if facts['contents'] == self.module.params['policy']:
                self.module.exit_json(changed=False, before=facts, after=facts)
            if self.module.check_mode:
                self.module.exit_json(changed=True, before=facts, after=facts)
            resp, info = api_request(module=self.module, endpoint='system/acl/%s.aclpolicy' % self.module.params['name'], method='PUT', data={'contents': self.module.params['policy']})
            if info['status'] == 200:
                self.module.exit_json(changed=True, before=facts, after=self.get_acl())
            elif info['status'] == 400:
                self.module.fail_json(msg="Unable to validate acl %s. Please ensure it's a valid ACL" % self.module.params['name'])
            elif info['status'] == 404:
                self.module.fail_json(msg="ACL %s doesn't exists. Cannot update." % self.module.params['name'])

    def remove_acl(self):
        facts = self.get_acl()
        if facts is None:
            self.module.exit_json(changed=False, before={}, after={})
        elif not self.module.check_mode:
            api_request(module=self.module, endpoint='system/acl/%s.aclpolicy' % self.module.params['name'], method='DELETE')
            self.module.exit_json(changed=True, before=facts, after={})