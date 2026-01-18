from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import (PyVmomi, find_dvs_by_name, find_dvspg_by_name)
class VmwareTagManager(VmwareRestClient):

    def __init__(self, module):
        """
        Constructor
        """
        super(VmwareTagManager, self).__init__(module)
        self.pyv = PyVmomi(module=module)
        moid = self.params.get('moid')
        self.object_type = self.params.get('object_type')
        managed_object_id = None
        if moid is not None:
            managed_object_id = moid
        else:
            object_name = self.params.get('object_name')
            managed_object = self.get_managed_object(object_name, self.object_type)
            if managed_object is None:
                self.module.fail_json(msg='Failed to find the managed object for %s with type %s' % (object_name, self.object_type))
            if not hasattr(managed_object, '_moId'):
                self.module.fail_json(msg='Unable to find managed object id for %s managed object' % object_name)
            managed_object_id = managed_object._moId
        self.dynamic_managed_object = DynamicID(type=self.object_type, id=managed_object_id)
        self.tag_service = self.api_client.tagging.Tag
        self.category_service = self.api_client.tagging.Category
        self.tag_association_svc = self.api_client.tagging.TagAssociation
        self.tag_names = self.params.get('tag_names')

    def get_managed_object(self, object_name=None, object_type=None):
        managed_object = None
        if not all([object_type, object_name]):
            return managed_object
        if object_type == 'VirtualMachine':
            managed_object = self.pyv.get_vm_or_template(object_name)
        if object_type == 'Folder':
            managed_object = self.pyv.find_folder_by_name(object_name)
        if object_type == 'Datacenter':
            managed_object = self.pyv.find_datacenter_by_name(object_name)
        if object_type == 'Datastore':
            managed_object = self.pyv.find_datastore_by_name(object_name)
        if object_type == 'DatastoreCluster':
            managed_object = self.pyv.find_datastore_cluster_by_name(object_name)
            self.object_type = 'StoragePod'
        if object_type == 'ClusterComputeResource':
            managed_object = self.pyv.find_cluster_by_name(object_name)
        if object_type == 'ResourcePool':
            managed_object = self.pyv.find_resource_pool_by_name(object_name)
        if object_type == 'HostSystem':
            managed_object = self.pyv.find_hostsystem_by_name(object_name)
        if object_type == 'DistributedVirtualSwitch':
            managed_object = find_dvs_by_name(self.pyv.content, object_name)
            self.object_type = 'VmwareDistributedVirtualSwitch'
        if object_type == 'DistributedVirtualPortgroup':
            dvs_name, pg_name = object_name.split(':', 1)
            dv_switch = find_dvs_by_name(self.pyv.content, dvs_name)
            if dv_switch is None:
                self.module.fail_json(msg='A distributed virtual switch with name %s does not exist' % dvs_name)
            managed_object = find_dvspg_by_name(dv_switch, pg_name)
        return managed_object

    def ensure_state(self):
        """
        Manage the internal state of tags

        """
        results = dict(changed=False, tag_status=dict())
        desired_tag_objs = set()
        changed = False
        action = self.params.get('state')
        try:
            current_tag_objs = self.get_tags_for_object(tag_service=self.tag_service, tag_assoc_svc=self.tag_association_svc, dobj=self.dynamic_managed_object, tags=set())
        except Error as error:
            self.module.fail_json(msg='%s' % self.get_error_message(error))
        results['tag_status']['previous_tags'] = ['%s:%s' % (tag_obj.category_id, tag_obj.name) for tag_obj in current_tag_objs]
        results['tag_status']['attached_tags'] = []
        results['tag_status']['detached_tags'] = []
        for tag in self.tag_names:
            category_obj, category_name, tag_name = (None, None, None)
            if isinstance(tag, dict):
                tag_name = tag.get('tag')
                category_name = tag.get('category')
                if category_name is not None:
                    category_obj = self.search_svc_object_by_name(self.category_service, category_name)
                    if category_obj is None:
                        self.module.fail_json(msg='Unable to find the category %s' % category_name)
            elif isinstance(tag, str):
                if ':' in tag:
                    category_name, tag_name = tag.split(':', 1)
                    category_obj = self.search_svc_object_by_name(self.category_service, category_name)
                    if category_obj is None:
                        self.module.fail_json(msg='Unable to find the category %s' % category_name)
                else:
                    tag_name = tag
            if category_obj is not None:
                tag_obj = self.get_tag_by_category_id(tag_name=tag_name, category_id=category_obj.id)
            else:
                tag_obj = self.get_tag_by_name(tag_name=tag_name)
            if tag_obj is None:
                self.module.fail_json(msg='Unable to find the tag %s' % tag_name)
            desired_tag_objs.add(tag_obj)
        detached_tag_objs = set()
        attached_tag_objs = set()
        if action in ('add', 'present'):
            tag_objs_to_attach = desired_tag_objs.difference(current_tag_objs)
            tag_ids_to_attach = [tag_obj.id for tag_obj in tag_objs_to_attach]
            if len(tag_ids_to_attach) > 0:
                try:
                    self.tag_association_svc.attach_multiple_tags_to_object(object_id=self.dynamic_managed_object, tag_ids=tag_ids_to_attach)
                    attached_tag_objs.update(tag_objs_to_attach)
                    current_tag_objs.update(tag_objs_to_attach)
                    changed = True
                except Error as error:
                    self.module.fail_json(msg='%s' % self.get_error_message(error))
        elif action == 'set':
            tag_objs_to_detach = current_tag_objs.difference(desired_tag_objs)
            tag_ids_to_detach = [tag_obj.id for tag_obj in tag_objs_to_detach]
            if len(tag_ids_to_detach) > 0:
                try:
                    self.tag_association_svc.detach_multiple_tags_from_object(object_id=self.dynamic_managed_object, tag_ids=tag_ids_to_detach)
                    detached_tag_objs.update(tag_objs_to_detach)
                    current_tag_objs.difference_update(tag_objs_to_detach)
                    changed = True
                except Error as error:
                    self.module.fail_json(msg='%s' % self.get_error_message(error))
            tag_objs_to_attach = desired_tag_objs.difference(current_tag_objs)
            tag_ids_to_attach = [tag_obj.id for tag_obj in tag_objs_to_attach]
            if len(tag_ids_to_attach) > 0:
                try:
                    self.tag_association_svc.attach_multiple_tags_to_object(object_id=self.dynamic_managed_object, tag_ids=tag_ids_to_attach)
                    attached_tag_objs.update(tag_objs_to_attach)
                    current_tag_objs.update(tag_objs_to_attach)
                    changed = True
                except Error as error:
                    self.module.fail_json(msg='%s' % self.get_error_message(error))
        elif action in ('remove', 'absent'):
            tag_objs_to_detach = current_tag_objs.intersection(desired_tag_objs)
            tag_ids_to_detach = [tag_obj.id for tag_obj in tag_objs_to_detach]
            if len(tag_ids_to_detach) > 0:
                try:
                    self.tag_association_svc.detach_multiple_tags_from_object(object_id=self.dynamic_managed_object, tag_ids=tag_ids_to_detach)
                    detached_tag_objs.update(tag_objs_to_detach)
                    current_tag_objs.difference_update(tag_objs_to_detach)
                    changed = True
                except Error as error:
                    self.module.fail_json(msg='%s' % self.get_error_message(error))
        results['tag_status']['detached_tags'] = ['%s:%s' % (tag_obj.category_id, tag_obj.name) for tag_obj in detached_tag_objs]
        results['tag_status']['attached_tags'] = ['%s:%s' % (tag_obj.category_id, tag_obj.name) for tag_obj in attached_tag_objs]
        results['tag_status']['current_tags'] = ['%s:%s' % (tag_obj.category_id, tag_obj.name) for tag_obj in current_tag_objs]
        results['changed'] = changed
        self.module.exit_json(**results)