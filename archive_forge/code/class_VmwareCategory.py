from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware import connect_to_api
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwareCategory(VmwareRestClient):

    def __init__(self, module):
        super(VmwareCategory, self).__init__(module)
        self.category_service = self.api_client.tagging.Category
        self.global_categories = dict()
        self.category_name = self.params.get('category_name')
        self.get_all_categories()
        self.content = connect_to_api(self.module, return_si=False)

    def ensure_state(self):
        """Manage internal states of categories. """
        desired_state = self.params.get('state')
        states = {'present': {'present': self.state_update_category, 'absent': self.state_create_category}, 'absent': {'present': self.state_delete_category, 'absent': self.state_unchanged}}
        states[desired_state][self.check_category_status()]()

    def state_create_category(self):
        """Create category."""
        category_spec = self.category_service.CreateSpec()
        category_spec.name = self.category_name
        category_spec.description = self.params.get('category_description')
        if self.params.get('category_cardinality') == 'single':
            category_spec.cardinality = CategoryModel.Cardinality.SINGLE
        else:
            category_spec.cardinality = CategoryModel.Cardinality.MULTIPLE
        associable_object_types = self.params.get('associable_object_types')

        def append_namespace(object_name):
            return '%s:%s' % (XMLNS_VMODL_BASE, object_name)
        associable_data = {'cluster': append_namespace('ClusterComputeResource'), 'datastore': append_namespace('Datastore'), 'datastore cluster': append_namespace('StoragePod'), 'folder': append_namespace('Folder'), 'host': append_namespace('HostSystem'), 'library item': append_namespace('com.vmware.content.library.Item'), 'datacenter': 'Datacenter', 'distributed port group': 'DistributedVirtualPortgroup', 'distributed switch': ['VmwareDistributedVirtualSwitch', 'DistributedVirtualSwitch'], 'content library': 'com.vmware.content.Library', 'resource pool': 'ResourcePool', 'vapp': 'VirtualApp', 'virtual machine': 'VirtualMachine', 'network': ['Network', 'HostNetwork', 'OpaqueNetwork'], 'host network': 'HostNetwork', 'opaque network': 'OpaqueNetwork'}
        obj_types_set = []
        if associable_object_types:
            for obj_type in associable_object_types:
                lower_obj_type = obj_type.lower()
                if lower_obj_type == 'all objects':
                    if LooseVersion(self.content.about.version) < LooseVersion('7'):
                        break
                    for category in list(associable_data.values()):
                        if isinstance(category, list):
                            obj_types_set.extend(category)
                        else:
                            obj_types_set.append(category)
                    break
                if lower_obj_type in associable_data:
                    value = associable_data.get(lower_obj_type)
                    if isinstance(value, list):
                        obj_types_set.extend(value)
                    else:
                        obj_types_set.append(value)
                else:
                    obj_types_set.append(obj_type)
        category_spec.associable_types = set(obj_types_set)
        category_id = ''
        try:
            category_id = self.category_service.create(category_spec)
        except Error as error:
            self.module.fail_json(msg='%s' % self.get_error_message(error))
        msg = 'No category created'
        changed = False
        if category_id:
            changed = True
            msg = "Category '%s' created." % category_spec.name
        self.module.exit_json(changed=changed, category_results=dict(msg=msg, category_id=category_id))

    def state_unchanged(self):
        """Return unchanged state."""
        self.module.exit_json(changed=False)

    def state_update_category(self):
        """Update category."""
        category_id = self.global_categories[self.category_name]['category_id']
        changed = False
        results = dict(msg='Category %s is unchanged.' % self.category_name, category_id=category_id)
        category_update_spec = self.category_service.UpdateSpec()
        change_list = []
        old_cat_desc = self.global_categories[self.category_name]['category_description']
        new_cat_desc = self.params.get('category_description')
        if new_cat_desc and new_cat_desc != old_cat_desc:
            category_update_spec.description = new_cat_desc
            results['msg'] = 'Category %s updated.' % self.category_name
            change_list.append(True)
        new_cat_name = self.params.get('new_category_name')
        if new_cat_name in self.global_categories:
            self.module.fail_json(msg='Unable to rename %s as %s already exists in configuration.' % (self.category_name, new_cat_name))
        old_cat_name = self.global_categories[self.category_name]['category_name']
        if new_cat_name and new_cat_name != old_cat_name:
            category_update_spec.name = new_cat_name
            results['msg'] = 'Category %s updated.' % self.category_name
            change_list.append(True)
        if any(change_list):
            try:
                self.category_service.update(category_id, category_update_spec)
                changed = True
            except Error as error:
                self.module.fail_json(msg='%s' % self.get_error_message(error))
        self.module.exit_json(changed=changed, category_results=results)

    def state_delete_category(self):
        """Delete category."""
        category_id = self.global_categories[self.category_name]['category_id']
        try:
            self.category_service.delete(category_id=category_id)
        except Error as error:
            self.module.fail_json(msg='%s' % self.get_error_message(error))
        self.module.exit_json(changed=True, category_results=dict(msg="Category '%s' deleted." % self.category_name, category_id=category_id))

    def check_category_status(self):
        """
        Check if category exists or not
        Returns: 'present' if category found, else 'absent'

        """
        if self.category_name in self.global_categories:
            return 'present'
        return 'absent'

    def get_all_categories(self):
        """Retrieve all category information."""
        try:
            for category in self.category_service.list():
                category_obj = self.category_service.get(category)
                self.global_categories[category_obj.name] = dict(category_description=category_obj.description, category_used_by=category_obj.used_by, category_cardinality=str(category_obj.cardinality), category_associable_types=category_obj.associable_types, category_id=category_obj.id, category_name=category_obj.name)
        except Error as error:
            self.module.fail_json(msg=self.get_error_message(error))
        except Exception as exc_err:
            self.module.fail_json(msg=to_native(exc_err))