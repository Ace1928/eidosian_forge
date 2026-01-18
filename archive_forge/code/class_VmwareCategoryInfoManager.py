from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwareCategoryInfoManager(VmwareRestClient):

    def __init__(self, module):
        super(VmwareCategoryInfoManager, self).__init__(module)
        self.category_service = self.api_client.tagging.Category

    def get_all_tag_categories(self):
        """Retrieve all tag category information."""
        global_tag_categories = []
        for category in self.category_service.list():
            category_obj = self.category_service.get(category)
            global_tag_categories.append(dict(category_description=category_obj.description, category_used_by=category_obj.used_by, category_cardinality=str(category_obj.cardinality), category_associable_types=category_obj.associable_types, category_id=category_obj.id, category_name=category_obj.name))
        self.module.exit_json(changed=False, tag_category_info=global_tag_categories)