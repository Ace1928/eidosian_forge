from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMCdnendpoint(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']), started=dict(type='bool'), purge=dict(type='bool'), purge_content_paths=dict(type='list', elements='str', default=['/']), profile_name=dict(type='str', required=True), origins=dict(type='list', elements='dict', options=origin_spec), origin_host_header=dict(type='str'), origin_path=dict(type='str'), content_types_to_compress=dict(type='list', elements='str'), is_compression_enabled=dict(type='bool', default=False), is_http_allowed=dict(type='bool', default=True), is_https_allowed=dict(type='bool', default=True), query_string_caching_behavior=dict(type='str', choices=['ignore_query_string', 'bypass_caching', 'use_query_string', 'not_set'], default='ignore_query_string'))
        self.resource_group = None
        self.name = None
        self.state = None
        self.started = None
        self.purge = None
        self.purge_content_paths = None
        self.location = None
        self.profile_name = None
        self.origins = None
        self.tags = None
        self.origin_host_header = None
        self.origin_path = None
        self.content_types_to_compress = None
        self.is_compression_enabled = None
        self.is_http_allowed = None
        self.is_https_allowed = None
        self.query_string_caching_behavior = None
        self.cdn_client = None
        self.results = dict(changed=False)
        super(AzureRMCdnendpoint, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.cdn_client = self.get_cdn_client()
        to_be_updated = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        if self.query_string_caching_behavior:
            self.query_string_caching_behavior = _snake_to_camel(self.query_string_caching_behavior, capitalize_first=True)
        response = self.get_cdnendpoint()
        if self.state == 'present':
            if not response:
                if self.started is None:
                    if self.origins is None:
                        self.fail('Origins is not provided when trying to create endpoint')
                    self.log('Need to create the Azure CDN endpoint')
                    if not self.check_mode:
                        result = self.create_cdnendpoint()
                        self.results['id'] = result['id']
                        self.results['host_name'] = result['host_name']
                        self.log('Creation done')
                    self.results['changed'] = True
                    return self.results
                else:
                    self.log("Can't stop/stop a non-existed endpoint")
                    self.fail('This endpoint is not found, stop/start is forbidden')
            else:
                self.log('Results : {0}'.format(response))
                self.results['id'] = response['id']
                self.results['host_name'] = response['host_name']
                update_tags, response['tags'] = self.update_tags(response['tags'])
                if update_tags:
                    to_be_updated = True
                if response['provisioning_state'] == 'Succeeded':
                    if self.started is False and response['resource_state'] == 'Running':
                        self.log('Need to stop the Azure CDN endpoint')
                        if not self.check_mode:
                            result = self.stop_cdnendpoint()
                            self.log('Endpoint stopped')
                        self.results['changed'] = True
                    elif self.started and response['resource_state'] == 'Stopped':
                        self.log('Need to start the Azure CDN endpoint')
                        if not self.check_mode:
                            result = self.start_cdnendpoint()
                            self.log('Endpoint started')
                        self.results['changed'] = True
                    elif self.started is not None:
                        self.module.warn('Start/Stop not performed due to current resource state {0}'.format(response['resource_state']))
                        self.results['changed'] = False
                    if self.purge:
                        self.log('Need to purge endpoint')
                        if not self.check_mode:
                            result = self.purge_cdnendpoint()
                            self.log('Endpoint purged')
                        self.results['changed'] = True
                    to_be_updated = to_be_updated or self.check_update(response)
                    if to_be_updated:
                        self.log('Need to update the Azure CDN endpoint')
                        self.results['changed'] = True
                        if not self.check_mode:
                            result = self.update_cdnendpoint()
                            self.results['host_name'] = result['host_name']
                            self.log('Update done')
                elif self.started is not None:
                    self.module.warn('Start/Stop not performed due to current provisioning state {0}'.format(response['provisioning_state']))
                    self.results['changed'] = False
        elif self.state == 'absent' and response:
            self.log('Need to delete the Azure CDN endpoint')
            self.results['changed'] = True
            if not self.check_mode:
                self.delete_cdnendpoint()
                self.log('Azure CDN endpoint deleted')
        return self.results

    def create_cdnendpoint(self):
        """
        Creates a Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint instance state dictionary
        """
        self.log('Creating the Azure CDN endpoint instance {0}'.format(self.name))
        origins = []
        for item in self.origins:
            origins.append(DeepCreatedOrigin(name=item['name'], host_name=item['host_name'], http_port=item['http_port'] if 'http_port' in item else None, https_port=item['https_port'] if 'https_port' in item else None))
        parameters = Endpoint(origins=origins, location=self.location, tags=self.tags, origin_host_header=self.origin_host_header, origin_path=self.origin_path, content_types_to_compress=default_content_types() if self.is_compression_enabled and (not self.content_types_to_compress) else self.content_types_to_compress, is_compression_enabled=self.is_compression_enabled if self.is_compression_enabled is not None else False, is_http_allowed=self.is_http_allowed if self.is_http_allowed is not None else True, is_https_allowed=self.is_https_allowed if self.is_https_allowed is not None else True, query_string_caching_behavior=self.query_string_caching_behavior if self.query_string_caching_behavior else QueryStringCachingBehavior.ignore_query_string)
        try:
            poller = self.cdn_client.endpoints.begin_create(self.resource_group, self.profile_name, self.name, parameters)
            response = self.get_poller_result(poller)
            return cdnendpoint_to_dict(response)
        except Exception as exc:
            self.log('Error attempting to create Azure CDN endpoint instance.')
            self.fail('Error creating Azure CDN endpoint instance: {0}'.format(exc.message))

    def update_cdnendpoint(self):
        """
        Updates a Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint instance state dictionary
        """
        self.log('Updating the Azure CDN endpoint instance {0}'.format(self.name))
        endpoint_update_properties = EndpointUpdateParameters(tags=self.tags, origin_host_header=self.origin_host_header, origin_path=self.origin_path, content_types_to_compress=default_content_types() if self.is_compression_enabled and (not self.content_types_to_compress) else self.content_types_to_compress, is_compression_enabled=self.is_compression_enabled, is_http_allowed=self.is_http_allowed, is_https_allowed=self.is_https_allowed, query_string_caching_behavior=self.query_string_caching_behavior)
        try:
            poller = self.cdn_client.endpoints.begin_update(self.resource_group, self.profile_name, self.name, endpoint_update_properties)
            response = self.get_poller_result(poller)
            return cdnendpoint_to_dict(response)
        except Exception as exc:
            self.log('Error attempting to update Azure CDN endpoint instance.')
            self.fail('Error updating Azure CDN endpoint instance: {0}'.format(exc.message))

    def delete_cdnendpoint(self):
        """
        Deletes the specified Azure CDN endpoint in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Azure CDN endpoint {0}'.format(self.name))
        try:
            poller = self.cdn_client.endpoints.begin_delete(self.resource_group, self.profile_name, self.name)
            self.get_poller_result(poller)
            return True
        except Exception as e:
            self.log('Error attempting to delete the Azure CDN endpoint.')
            self.fail('Error deleting the Azure CDN endpoint: {0}'.format(e.message))
            return False

    def get_cdnendpoint(self):
        """
        Gets the properties of the specified Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint state dictionary
        """
        self.log('Checking if the Azure CDN endpoint {0} is present'.format(self.name))
        try:
            response = self.cdn_client.endpoints.get(self.resource_group, self.profile_name, self.name)
            self.log('Response : {0}'.format(response))
            self.log('Azure CDN endpoint : {0} found'.format(response.name))
            return cdnendpoint_to_dict(response)
        except Exception:
            self.log('Did not find the Azure CDN endpoint.')
            return False

    def start_cdnendpoint(self):
        """
        Starts an existing Azure CDN endpoint that is on a stopped state.

        :return: deserialized Azure CDN endpoint state dictionary
        """
        self.log('Starting the Azure CDN endpoint {0}'.format(self.name))
        try:
            poller = self.cdn_client.endpoints.begin_start(self.resource_group, self.profile_name, self.name)
            response = self.get_poller_result(poller)
            self.log('Response : {0}'.format(response))
            self.log('Azure CDN endpoint : {0} started'.format(response.name))
            return self.get_cdnendpoint()
        except Exception:
            self.log('Fail to start the Azure CDN endpoint.')
            return False

    def purge_cdnendpoint(self):
        """
        Purges an existing Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint state dictionary
        """
        self.log('Purging the Azure CDN endpoint {0}'.format(self.name))
        try:
            poller = self.cdn_client.endpoints.begin_purge_content(self.resource_group, self.profile_name, self.name, content_file_paths=dict(content_paths=self.purge_content_paths))
            response = self.get_poller_result(poller)
            self.log('Response : {0}'.format(response))
            return self.get_cdnendpoint()
        except Exception as e:
            self.fail('Fail to purge the Azure CDN endpoint.')
            return False

    def stop_cdnendpoint(self):
        """
        Stops an existing Azure CDN endpoint that is on a running state.

        :return: deserialized Azure CDN endpoint state dictionary
        """
        self.log('Stopping the Azure CDN endpoint {0}'.format(self.name))
        try:
            poller = self.cdn_client.endpoints.begin_stop(self.resource_group, self.profile_name, self.name)
            response = self.get_poller_result(poller)
            self.log('Response : {0}'.format(response))
            self.log('Azure CDN endpoint : {0} stopped'.format(response.name))
            return self.get_cdnendpoint()
        except Exception:
            self.log('Fail to stop the Azure CDN endpoint.')
            return False

    def check_update(self, response):
        if self.origin_host_header and response['origin_host_header'] != self.origin_host_header:
            self.log('Origin host header Diff - Origin {0} / Update {1}'.format(response['origin_host_header'], self.origin_host_header))
            return True
        if self.origin_path and response['origin_path'] != self.origin_path:
            self.log('Origin path Diff - Origin {0} / Update {1}'.format(response['origin_path'], self.origin_path))
            return True
        if self.content_types_to_compress and response['content_types_to_compress'] != self.content_types_to_compress:
            self.log('Content types to compress Diff - Origin {0} / Update {1}'.format(response['content_types_to_compress'], self.content_types_to_compress))
            return True
        if self.is_compression_enabled is not None and response['is_compression_enabled'] != self.is_compression_enabled:
            self.log('is_compression_enabled Diff - Origin {0} / Update {1}'.format(response['is_compression_enabled'], self.is_compression_enabled))
            return True
        if self.is_http_allowed is not None and response['is_http_allowed'] != self.is_http_allowed:
            self.log('is_http_allowed Diff - Origin {0} / Update {1}'.format(response['is_http_allowed'], self.is_http_allowed))
            return True
        if self.is_https_allowed is not None and response['is_https_allowed'] != self.is_https_allowed:
            self.log('is_https_allowed Diff - Origin {0} / Update {1}'.format(response['is_https_allowed'], self.is_https_allowed))
            return True
        if self.query_string_caching_behavior and _snake_to_camel(response['query_string_caching_behavior']).lower() != _snake_to_camel(self.query_string_caching_behavior).lower():
            self.log('query_string_caching_behavior Diff - Origin {0} / Update {1}'.format(response['query_string_caching_behavior'], self.query_string_caching_behavior))
            return True
        return False

    def get_cdn_client(self):
        if not self.cdn_client:
            self.cdn_client = self.get_mgmt_svc_client(CdnManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2017-04-02')
        return self.cdn_client