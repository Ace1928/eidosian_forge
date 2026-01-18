from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmTagInfoManager(VmwareRestClient):

    def __init__(self, module):
        """Constructor."""
        super(VmTagInfoManager, self).__init__(module)

    def get_all_tags(self):
        """
        Retrieve all tag information.
        """
        global_tag_info = list()
        global_tags = dict()
        tag_service = self.api_client.tagging.Tag
        for tag in tag_service.list():
            tag_obj = tag_service.get(tag)
            global_tags[tag_obj.name] = dict(tag_description=tag_obj.description, tag_used_by=tag_obj.used_by, tag_category_id=tag_obj.category_id, tag_id=tag_obj.id)
            global_tag_info.append(dict(tag_name=tag_obj.name, tag_description=tag_obj.description, tag_used_by=tag_obj.used_by, tag_category_id=tag_obj.category_id, tag_id=tag_obj.id))
        self.module.exit_json(changed=False, tag_facts=global_tags, tag_info=global_tag_info)