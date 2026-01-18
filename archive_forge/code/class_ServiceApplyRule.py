from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec
from ansible_collections.t_systems_mms.icinga_director.plugins.module_utils.icinga import (
class ServiceApplyRule(Icinga2APIObject):

    def __init__(self, module, data):
        path = '/service'
        super(ServiceApplyRule, self).__init__(module, path, data)

    def exists(self):
        ret = self.call_url(path='/serviceapplyrules')
        if ret['code'] == 200:
            for existing_rule in ret['data']['objects']:
                if existing_rule['object_name'] == self.data['object_name']:
                    self.object_id = existing_rule['id']
                    return self.object_id
        return False

    def delete(self):
        return super(ServiceApplyRule, self).delete(find_by='id')

    def modify(self):
        return super(ServiceApplyRule, self).modify(find_by='id')

    def diff(self):
        return super(ServiceApplyRule, self).diff(find_by='id')