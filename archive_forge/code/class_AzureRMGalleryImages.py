from __future__ import absolute_import, division, print_function
import time
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMGalleryImages(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', updatable=False, disposition='resourceGroupName', required=True), gallery_name=dict(type='str', updatable=False, disposition='galleryName', required=True), name=dict(type='str', updatable=False, disposition='galleryImageName', required=True), location=dict(type='str', updatable=False, disposition='/'), description=dict(type='str', disposition='/properties/*'), eula=dict(type='str', disposition='/properties/*'), privacy_statement_uri=dict(type='str', disposition='/properties/privacyStatementUri'), release_note_uri=dict(type='str', disposition='/properties/releaseNoteUri'), os_type=dict(type='str', disposition='/properties/osType', choices=['windows', 'linux']), os_state=dict(type='str', disposition='/properties/osState', choices=['generalized', 'specialized']), hypervgeneration=dict(type='str', disposition='/properties/hyperVGeneration', choices=['V1', 'V2']), end_of_life_date=dict(type='str', disposition='/properties/endOfLifeDate'), identifier=dict(type='dict', disposition='/properties/*', options=dict(publisher=dict(type='str', required=True, updatable=False), offer=dict(type='str', required=True), sku=dict(type='str', required=True))), recommended=dict(type='dict', disposition='/properties/*', options=dict(v_cpus=dict(type='dict', disposition='vCPUs', options=dict(min=dict(type='int'), max=dict(type='int'))), memory=dict(type='dict', options=dict(min=dict(type='int'), max=dict(type='int'))))), disallowed=dict(type='dict', disposition='/properties/*', options=dict(disk_types=dict(type='list', elements='str', disposition='diskTypes'))), purchase_plan=dict(type='dict', disposition='/properties/purchasePlan', options=dict(name=dict(type='str'), publisher=dict(type='str'), product=dict(type='str'))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.gallery_name = None
        self.name = None
        self.gallery_image = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2019-07-01'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMGalleryImages, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.body:
            self.body['location'] = resource_group.location
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.Compute' + '/galleries' + '/{{ gallery_name }}' + '/images' + '/{{ image_name }}'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ gallery_name }}', self.gallery_name)
        self.url = self.url.replace('{{ image_name }}', self.name)
        old_response = self.get_resource()
        if not old_response:
            self.log("GalleryImage instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('GalleryImage instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                modifiers = {}
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                self.results['compare'] = []
                if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the GalleryImage instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_resource()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('GalleryImage instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
            while self.get_resource():
                time.sleep(20)
        else:
            self.log('GalleryImage instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
        return self.results

    def create_update_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as exc:
            self.log('Error attempting to create the GalleryImage instance.')
            self.fail('Error creating the GalleryImage instance: {0}'.format(str(exc)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def delete_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete the GalleryImage instance.')
            self.fail('Error deleting the GalleryImage instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            response = json.loads(response.body())
            found = True
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Did not find the AzureFirewall instance.')
        if found is True:
            return response
        return False