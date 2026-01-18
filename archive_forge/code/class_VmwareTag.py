from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwareTag(VmwareRestClient):

    def __init__(self, module):
        super(VmwareTag, self).__init__(module)
        self.tag_service = self.api_client.tagging.Tag
        self.tag_association_svc = self.api_client.tagging.TagAssociation