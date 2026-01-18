from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def add_file_security_permissions_acl(self, acl):
    api = 'protocols/file-security/permissions/%s/%s/acl' % (self.svm_uuid, self.url_encode(self.parameters['path']))
    for option in ('access_control', 'propagation_mode'):
        self.set_option(acl, option)
    dummy, error = rest_generic.post_async(self.rest_api, api, acl, timeout=0)
    if error:
        self.module.fail_json(msg='Error adding file security permissions acl %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())