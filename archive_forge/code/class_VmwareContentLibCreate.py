from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
class VmwareContentLibCreate(VmwareRestClient):

    def __init__(self, module):
        """Constructor."""
        super(VmwareContentLibCreate, self).__init__(module)
        self.content_service = self.api_client
        self.local_libraries = dict()
        self.existing_library_names = []
        self.library_name = self.params.get('library_name')
        self.library_description = self.params.get('library_description')
        self.library_type = self.params.get('library_type')
        self.library_types = dict()
        self.subscription_url = self.params.get('subscription_url')
        self.ssl_thumbprint = self.params.get('ssl_thumbprint')
        self.datastore_name = self.params.get('datastore_name')
        self.update_on_demand = self.params.get('update_on_demand')
        self.library_types = {'local': self.content_service.content.LocalLibrary, 'subscribed': self.content_service.content.SubscribedLibrary}
        self.get_all_libraries(self.library_types['local'])
        self.get_all_libraries(self.library_types['subscribed'])
        self.library_service = self.library_types[self.library_type]
        self.pyv = PyVmomi(module=module)

    def process_state(self):
        """
        Manage states of Content Library
        """
        self.desired_state = self.params.get('state')
        library_states = {'absent': {'present': self.state_destroy_library, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_update_library, 'absent': self.state_create_library}}
        library_states[self.desired_state][self.check_content_library_status()]()

    def get_all_libraries(self, library_service):
        content_libs = library_service.list()
        if content_libs:
            for content_lib in content_libs:
                lib_details = library_service.get(content_lib)
                lib_dict = dict(lib_name=lib_details.name, lib_description=lib_details.description, lib_id=lib_details.id, lib_type=lib_details.type)
                if lib_details.type == 'SUBSCRIBED':
                    lib_dict['lib_sub_url'] = lib_details.subscription_info.subscription_url
                    lib_dict['lib_sub_on_demand'] = lib_details.subscription_info.on_demand
                    lib_dict['lib_sub_ssl_thumbprint'] = lib_details.subscription_info.ssl_thumbprint
                self.local_libraries[lib_details.name] = lib_dict
                self.existing_library_names.append(lib_details.name)

    def check_content_library_status(self):
        """
        Check if Content Library exists or not
        Returns: 'present' if library found, else 'absent'

        """
        ret = 'present' if self.library_name in self.local_libraries else 'absent'
        return ret

    def fail_when_duplicated(self):
        if self.existing_library_names.count(self.library_name) > 1:
            self.module.fail_json(msg='Operation cannot continue, library [%s] is not unique' % self.library_name)

    def state_exit_unchanged(self):
        """
        Return unchanged state

        """
        self.module.exit_json(changed=False)

    def set_subscription_spec(self):
        if 'https:' in self.subscription_url and (not self.ssl_thumbprint):
            self.module.fail_json(msg='While using HTTPS, a SSL thumbprint must be provided.')
        subscription_info = SubscriptionInfo()
        subscription_info.on_demand = self.update_on_demand
        subscription_info.automatic_sync_enabled = True
        subscription_info.subscription_url = self.subscription_url
        if 'https:' in self.subscription_url:
            subscription_info.ssl_thumbprint = self.ssl_thumbprint
        return subscription_info

    def create_update(self, spec, library_id=None, update=False):
        """
        Create or update call and exit cleanly if call completes
        """
        if self.module.check_mode:
            action = 'would be updated' if update else 'would be created'
        else:
            try:
                if update:
                    self.library_service.update(library_id, spec)
                    action = 'updated'
                else:
                    library_id = self.library_service.create(create_spec=spec, client_token=str(uuid.uuid4()))
                    action = 'created'
            except ResourceInaccessible as e:
                message = 'vCenter Failed to make connection to %s with exception: %s If using HTTPS, check that the SSL thumbprint is valid' % (self.subscription_url, str(e))
                self.module.fail_json(msg=message)
        content_library_info = dict(msg="Content Library '%s' %s." % (spec.name, action), library_id=library_id, library_description=self.library_description, library_type=spec.type)
        if spec.type == 'SUBSCRIBED':
            content_library_info['library_subscription_url'] = spec.subscription_info.subscription_url
            content_library_info['library_subscription_on_demand'] = spec.subscription_info.on_demand
            content_library_info['library_subscription_ssl_thumbprint'] = spec.subscription_info.ssl_thumbprint
        self.module.exit_json(changed=True, content_library_info=content_library_info)

    def state_create_library(self):
        if not self.datastore_name:
            self.module.fail_json(msg='datastore_name must be specified for create operations')
        datastore_id = self.pyv.find_datastore_by_name(datastore_name=self.datastore_name)
        if not datastore_id:
            self.module.fail_json(msg='Failed to find the datastore %s' % self.datastore_name)
        self.datastore_id = datastore_id._moId
        storage_backings = []
        storage_backing = StorageBacking(type=StorageBacking.Type.DATASTORE, datastore_id=self.datastore_id)
        storage_backings.append(storage_backing)
        create_spec = LibraryModel()
        create_spec.name = self.library_name
        create_spec.description = self.library_description
        self.library_types = {'local': create_spec.LibraryType.LOCAL, 'subscribed': create_spec.LibraryType.SUBSCRIBED}
        create_spec.type = self.library_types[self.library_type]
        create_spec.storage_backings = storage_backings
        if self.library_type == 'subscribed':
            subscription_info = self.set_subscription_spec()
            subscription_info.authentication_method = SubscriptionInfo.AuthenticationMethod.NONE
            create_spec.subscription_info = subscription_info
        self.create_update(spec=create_spec)

    def state_update_library(self):
        """
        Update Content Library

        """
        self.fail_when_duplicated()
        changed = False
        library_id = self.local_libraries[self.library_name]['lib_id']
        library_update_spec = LibraryModel()
        existing_library_type = self.local_libraries[self.library_name]['lib_type'].lower()
        if existing_library_type != self.library_type:
            self.module.fail_json(msg='Library [%s] is of type %s, cannot be changed to %s' % (self.library_name, existing_library_type, self.library_type))
        if self.library_type == 'subscribed':
            existing_subscription_url = self.local_libraries[self.library_name]['lib_sub_url']
            sub_url_changed = existing_subscription_url != self.subscription_url
            existing_on_demand = self.local_libraries[self.library_name]['lib_sub_on_demand']
            sub_on_demand_changed = existing_on_demand != self.update_on_demand
            sub_ssl_thumbprint_changed = False
            if 'https:' in self.subscription_url and self.ssl_thumbprint:
                existing_ssl_thumbprint = self.local_libraries[self.library_name]['lib_sub_ssl_thumbprint']
                sub_ssl_thumbprint_changed = existing_ssl_thumbprint != self.ssl_thumbprint
            if sub_url_changed or sub_on_demand_changed or sub_ssl_thumbprint_changed:
                subscription_info = self.set_subscription_spec()
                library_update_spec.subscription_info = subscription_info
                changed = True
        library_desc = self.local_libraries[self.library_name]['lib_description']
        desired_lib_desc = self.params.get('library_description')
        if library_desc != desired_lib_desc:
            library_update_spec.description = desired_lib_desc
            changed = True
        if changed:
            library_update_spec.name = self.library_name
            self.create_update(spec=library_update_spec, library_id=library_id, update=True)
        content_library_info = dict(msg='Content Library %s is unchanged.' % self.library_name, library_id=library_id)
        self.module.exit_json(changed=False, content_library_info=dict(msg=content_library_info, library_id=library_id))

    def state_destroy_library(self):
        """
        Delete Content Library

        """
        self.fail_when_duplicated()
        library_id = self.local_libraries[self.library_name]['lib_id']
        library_service = self.library_types[self.local_libraries[self.library_name]['lib_type'].lower()]
        if self.module.check_mode:
            action = 'would be deleted'
        else:
            action = 'deleted'
            library_service.delete(library_id=library_id)
        self.module.exit_json(changed=True, content_library_info=dict(msg="Content Library '%s' %s." % (self.library_name, action), library_id=library_id))