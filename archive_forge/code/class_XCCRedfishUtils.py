from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
class XCCRedfishUtils(RedfishUtils):

    @staticmethod
    def _find_empty_virt_media_slot(resources, media_types, media_match_strict=True):
        for uri, data in resources.items():
            if 'MediaTypes' in data and media_types:
                if not set(media_types).intersection(set(data['MediaTypes'])):
                    continue
            elif media_match_strict:
                continue
            if 'RDOC' in uri:
                continue
            if 'Remote' in uri:
                continue
            if not data.get('Inserted', False) and (not data.get('ImageName')):
                return (uri, data)
        return (None, None)

    def virtual_media_eject_one(self, image_url):
        response = self.get_request(self.root_uri + self.systems_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        if 'VirtualMedia' not in data:
            response = self.get_request(self.root_uri + self.manager_uri)
            if response['ret'] is False:
                return response
            data = response['data']
            if 'VirtualMedia' not in data:
                return {'ret': False, 'msg': 'VirtualMedia resource not found'}
        virt_media_uri = data['VirtualMedia']['@odata.id']
        response = self.get_request(self.root_uri + virt_media_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        virt_media_list = []
        for member in data[u'Members']:
            virt_media_list.append(member[u'@odata.id'])
        resources, headers = self._read_virt_media_resources(virt_media_list)
        uri, data, eject = self._find_virt_media_to_eject(resources, image_url)
        if uri and eject:
            if 'Actions' not in data or '#VirtualMedia.EjectMedia' not in data['Actions']:
                h = headers[uri]
                if 'allow' in h:
                    methods = [m.strip() for m in h.get('allow').split(',')]
                    if 'PATCH' not in methods:
                        return {'ret': False, 'msg': '%s action not found and PATCH not allowed' % '#VirtualMedia.EjectMedia'}
                return self.virtual_media_eject_via_patch(uri)
            else:
                action = data['Actions']['#VirtualMedia.EjectMedia']
                if 'target' not in action:
                    return {'ret': False, 'msg': 'target URI property missing from Action #VirtualMedia.EjectMedia'}
                action_uri = action['target']
                payload = {}
                response = self.post_request(self.root_uri + action_uri, payload)
                if response['ret'] is False:
                    return response
                return {'ret': True, 'changed': True, 'msg': 'VirtualMedia ejected'}
        elif uri and (not eject):
            return {'ret': True, 'changed': False, 'msg': "VirtualMedia image '%s' already ejected" % image_url}
        else:
            return {'ret': False, 'changed': False, 'msg': "No VirtualMedia resource found with image '%s' inserted" % image_url}

    def virtual_media_eject(self, options):
        if options:
            image_url = options.get('image_url')
            if image_url:
                return self.virtual_media_eject_one(image_url)
        response = self.get_request(self.root_uri + self.systems_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        if 'VirtualMedia' not in data:
            response = self.get_request(self.root_uri + self.manager_uri)
            if response['ret'] is False:
                return response
            data = response['data']
            if 'VirtualMedia' not in data:
                return {'ret': False, 'msg': 'VirtualMedia resource not found'}
        virt_media_uri = data['VirtualMedia']['@odata.id']
        response = self.get_request(self.root_uri + virt_media_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        virt_media_list = []
        for member in data[u'Members']:
            virt_media_list.append(member[u'@odata.id'])
        resources, headers = self._read_virt_media_resources(virt_media_list)
        ejected_media_list = []
        for uri, data in resources.items():
            if data.get('Image') and data.get('Inserted', True):
                returndict = self.virtual_media_eject_one(data.get('Image'))
                if not returndict['ret']:
                    return returndict
                ejected_media_list.append(data.get('Image'))
        if len(ejected_media_list) == 0:
            return {'ret': True, 'changed': False, 'msg': 'No VirtualMedia image inserted'}
        else:
            return {'ret': True, 'changed': True, 'msg': 'VirtualMedia %s ejected' % str(ejected_media_list)}

    def virtual_media_insert(self, options):
        param_map = {'Inserted': 'inserted', 'WriteProtected': 'write_protected', 'UserName': 'username', 'Password': 'password', 'TransferProtocolType': 'transfer_protocol_type', 'TransferMethod': 'transfer_method'}
        image_url = options.get('image_url')
        if not image_url:
            return {'ret': False, 'msg': 'image_url option required for VirtualMediaInsert'}
        media_types = options.get('media_types')
        response = self.get_request(self.root_uri + self.systems_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        if 'VirtualMedia' not in data:
            response = self.get_request(self.root_uri + self.manager_uri)
            if response['ret'] is False:
                return response
            data = response['data']
            if 'VirtualMedia' not in data:
                return {'ret': False, 'msg': 'VirtualMedia resource not found'}
        virt_media_uri = data['VirtualMedia']['@odata.id']
        response = self.get_request(self.root_uri + virt_media_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        virt_media_list = []
        for member in data[u'Members']:
            virt_media_list.append(member[u'@odata.id'])
        resources, headers = self._read_virt_media_resources(virt_media_list)
        if self._virt_media_image_inserted(resources, image_url):
            return {'ret': True, 'changed': False, 'msg': "VirtualMedia '%s' already inserted" % image_url}
        uri, data = self._find_empty_virt_media_slot(resources, media_types, media_match_strict=True)
        if not uri:
            uri, data = self._find_empty_virt_media_slot(resources, media_types, media_match_strict=False)
        if not uri:
            return {'ret': False, 'msg': 'Unable to find an available VirtualMedia resource %s' % ('supporting ' + str(media_types) if media_types else '')}
        if 'Actions' not in data or '#VirtualMedia.InsertMedia' not in data['Actions']:
            h = headers[uri]
            if 'allow' in h:
                methods = [m.strip() for m in h.get('allow').split(',')]
                if 'PATCH' not in methods:
                    return {'ret': False, 'msg': '%s action not found and PATCH not allowed' % '#VirtualMedia.InsertMedia'}
            return self.virtual_media_insert_via_patch(options, param_map, uri, data)
        action = data['Actions']['#VirtualMedia.InsertMedia']
        if 'target' not in action:
            return {'ret': False, 'msg': 'target URI missing from Action #VirtualMedia.InsertMedia'}
        action_uri = action['target']
        ai = self._get_all_action_info_values(action)
        payload = self._insert_virt_media_payload(options, param_map, data, ai)
        response = self.post_request(self.root_uri + action_uri, payload)
        if response['ret'] is False:
            return response
        return {'ret': True, 'changed': True, 'msg': 'VirtualMedia inserted'}

    def raw_get_resource(self, resource_uri):
        if resource_uri is None:
            return {'ret': False, 'msg': 'resource_uri is missing'}
        response = self.get_request(self.root_uri + resource_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        return {'ret': True, 'data': data}

    def raw_get_collection_resource(self, resource_uri):
        if resource_uri is None:
            return {'ret': False, 'msg': 'resource_uri is missing'}
        response = self.get_request(self.root_uri + resource_uri)
        if response['ret'] is False:
            return response
        if 'Members' not in response['data']:
            return {'ret': False, 'msg': "Specified resource_uri doesn't have Members property"}
        member_list = [i['@odata.id'] for i in response['data'].get('Members', [])]
        data_list = []
        for member_uri in member_list:
            uri = self.root_uri + member_uri
            response = self.get_request(uri)
            if response['ret'] is False:
                return response
            data = response['data']
            data_list.append(data)
        return {'ret': True, 'data_list': data_list}

    def raw_patch_resource(self, resource_uri, request_body):
        if resource_uri is None:
            return {'ret': False, 'msg': 'resource_uri is missing'}
        if request_body is None:
            return {'ret': False, 'msg': 'request_body is missing'}
        response = self.get_request(self.root_uri + resource_uri)
        if response['ret'] is False:
            return response
        original_etag = response['data']['@odata.etag']
        data = response['data']
        for key in request_body.keys():
            if key not in data:
                return {'ret': False, 'msg': 'Key %s not found. Supported key list: %s' % (key, str(data.keys()))}
        response = self.patch_request(self.root_uri + resource_uri, request_body)
        if response['ret'] is False:
            return response
        current_etag = ''
        if 'data' in response and '@odata.etag' in response['data']:
            current_etag = response['data']['@odata.etag']
        if current_etag != original_etag:
            return {'ret': True, 'changed': True}
        else:
            return {'ret': True, 'changed': False}

    def raw_post_resource(self, resource_uri, request_body):
        if resource_uri is None:
            return {'ret': False, 'msg': 'resource_uri is missing'}
        resource_uri_has_actions = True
        if '/Actions/' not in resource_uri:
            resource_uri_has_actions = False
        if request_body is None:
            return {'ret': False, 'msg': 'request_body is missing'}
        action_base_uri = resource_uri.split('/Actions/')[0]
        response = self.get_request(self.root_uri + action_base_uri)
        if response['ret'] is False:
            return response
        if 'Actions' not in response['data']:
            if resource_uri_has_actions:
                return {'ret': False, 'msg': 'Actions property not found in %s' % action_base_uri}
            else:
                response['data']['Actions'] = {}
        action_found = False
        action_info_uri = None
        action_target_uri_list = []
        for key in response['data']['Actions'].keys():
            if action_found:
                break
            if not key.startswith('#'):
                continue
            if 'target' in response['data']['Actions'][key]:
                if resource_uri == response['data']['Actions'][key]['target']:
                    action_found = True
                    if '@Redfish.ActionInfo' in response['data']['Actions'][key]:
                        action_info_uri = response['data']['Actions'][key]['@Redfish.ActionInfo']
                else:
                    action_target_uri_list.append(response['data']['Actions'][key]['target'])
        if not action_found and 'Oem' in response['data']['Actions']:
            for key in response['data']['Actions']['Oem'].keys():
                if action_found:
                    break
                if not key.startswith('#'):
                    continue
                if 'target' in response['data']['Actions']['Oem'][key]:
                    if resource_uri == response['data']['Actions']['Oem'][key]['target']:
                        action_found = True
                        if '@Redfish.ActionInfo' in response['data']['Actions']['Oem'][key]:
                            action_info_uri = response['data']['Actions']['Oem'][key]['@Redfish.ActionInfo']
                    else:
                        action_target_uri_list.append(response['data']['Actions']['Oem'][key]['target'])
        if not action_found and resource_uri_has_actions:
            return {'ret': False, 'msg': 'Specified resource_uri is not a supported action target uri, please specify a supported target uri instead. Supported uri: %s' % str(action_target_uri_list)}
        if action_info_uri is not None:
            response = self.get_request(self.root_uri + action_info_uri)
            if response['ret'] is False:
                return response
            for key in request_body.keys():
                key_found = False
                for para in response['data']['Parameters']:
                    if key == para['Name']:
                        key_found = True
                        break
                if not key_found:
                    return {'ret': False, 'msg': 'Invalid property %s found in request_body. Please refer to @Redfish.ActionInfo Parameters: %s' % (key, str(response['data']['Parameters']))}
        response = self.post_request(self.root_uri + resource_uri, request_body)
        if response['ret'] is False:
            return response
        return {'ret': True, 'changed': True}