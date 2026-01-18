from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class LunMapping(object):

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(state=dict(required=True, choices=['present', 'absent']), target=dict(required=False, default=None), volume_name=dict(required=True, aliases=['volume']), lun=dict(type='int', required=False), target_type=dict(required=False, choices=['host', 'group'])))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        args = self.module.params
        self.state = args['state'] in ['present']
        self.target = args['target']
        self.volume = args['volume_name']
        self.lun = args['lun']
        self.target_type = args['target_type']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.check_mode = self.module.check_mode
        self.creds = dict(url_username=args['api_username'], url_password=args['api_password'], validate_certs=args['validate_certs'])
        self.mapping_info = None
        if not self.url.endswith('/'):
            self.url += '/'

    def update_mapping_info(self):
        """Collect the current state of the storage array."""
        response = None
        try:
            rc, response = request(self.url + 'storage-systems/%s/graph' % self.ssid, method='GET', headers=HEADERS, **self.creds)
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve storage array graph. Id [%s]. Error [%s]' % (self.ssid, to_native(error)))
        target_reference = {}
        target_name = {}
        target_type = {}
        if self.target_type is None or self.target_type == 'host':
            for host in response['storagePoolBundle']['host']:
                target_reference.update({host['hostRef']: host['name']})
                target_name.update({host['name']: host['hostRef']})
                target_type.update({host['name']: 'host'})
        if self.target_type is None or self.target_type == 'group':
            for cluster in response['storagePoolBundle']['cluster']:
                if self.target and self.target_type is None and (cluster['name'] == self.target) and (self.target in target_name.keys()):
                    self.module.fail_json(msg='Ambiguous target type: target name is used for both host and group targets! Id [%s]' % self.ssid)
                target_reference.update({cluster['clusterRef']: cluster['name']})
                target_name.update({cluster['name']: cluster['clusterRef']})
                target_type.update({cluster['name']: 'group'})
        volume_reference = {}
        volume_name = {}
        lun_name = {}
        for volume in response['volume']:
            volume_reference.update({volume['volumeRef']: volume['name']})
            volume_name.update({volume['name']: volume['volumeRef']})
            if volume['listOfMappings']:
                lun_name.update({volume['name']: volume['listOfMappings'][0]['lun']})
        for volume in response['highLevelVolBundle']['thinVolume']:
            volume_reference.update({volume['volumeRef']: volume['name']})
            volume_name.update({volume['name']: volume['volumeRef']})
            if volume['listOfMappings']:
                lun_name.update({volume['name']: volume['listOfMappings'][0]['lun']})
        self.mapping_info = dict(lun_mapping=[dict(volume_reference=mapping['volumeRef'], map_reference=mapping['mapRef'], lun_mapping_reference=mapping['lunMappingRef'], lun=mapping['lun']) for mapping in response['storagePoolBundle']['lunMapping']], volume_by_reference=volume_reference, volume_by_name=volume_name, lun_by_name=lun_name, target_by_reference=target_reference, target_by_name=target_name, target_type_by_name=target_type)

    def get_lun_mapping(self):
        """Find the matching lun mapping reference.

        Returns: tuple(bool, int, int): contains volume match, volume mapping reference and mapping lun
        """
        target_match = False
        reference = None
        lun = None
        self.update_mapping_info()
        if self.lun and any((self.lun == lun_mapping['lun'] and self.target == self.mapping_info['target_by_reference'][lun_mapping['map_reference']] and (self.volume != self.mapping_info['volume_by_reference'][lun_mapping['volume_reference']]) for lun_mapping in self.mapping_info['lun_mapping'])):
            self.module.fail_json(msg='Option lun value is already in use for target! Array Id [%s].' % self.ssid)
        if self.target and self.target_type and (self.target in self.mapping_info['target_type_by_name'].keys()) and (self.mapping_info['target_type_by_name'][self.target] != self.target_type):
            self.module.fail_json(msg='Option target does not match the specified target_type! Id [%s].' % self.ssid)
        if self.state:
            if self.volume not in self.mapping_info['volume_by_name'].keys():
                self.module.fail_json(msg='Volume does not exist. Id [%s].' % self.ssid)
            if self.target and self.target not in self.mapping_info['target_by_name'].keys():
                self.module.fail_json(msg="Target does not exist. Id [%s'." % self.ssid)
        for lun_mapping in self.mapping_info['lun_mapping']:
            if lun_mapping['volume_reference'] == self.mapping_info['volume_by_name'][self.volume]:
                reference = lun_mapping['lun_mapping_reference']
                lun = lun_mapping['lun']
                if lun_mapping['map_reference'] in self.mapping_info['target_by_reference'].keys() and self.mapping_info['target_by_reference'][lun_mapping['map_reference']] == self.target and (self.lun is None or lun == self.lun):
                    target_match = True
        return (target_match, reference, lun)

    def update(self):
        """Execute the changes the require changes on the storage array."""
        target_match, lun_reference, lun = self.get_lun_mapping()
        update = self.state and (not target_match) or (not self.state and target_match)
        if update and (not self.check_mode):
            try:
                if self.state:
                    body = dict()
                    target = None if not self.target else self.mapping_info['target_by_name'][self.target]
                    if target:
                        body.update(dict(targetId=target))
                    if self.lun is not None:
                        body.update(dict(lun=self.lun))
                    if lun_reference:
                        rc, response = request(self.url + 'storage-systems/%s/volume-mappings/%s/move' % (self.ssid, lun_reference), method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
                    else:
                        body.update(dict(mappableObjectId=self.mapping_info['volume_by_name'][self.volume]))
                        rc, response = request(self.url + 'storage-systems/%s/volume-mappings' % self.ssid, method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
                else:
                    rc, response = request(self.url + 'storage-systems/%s/volume-mappings/%s' % (self.ssid, lun_reference), method='DELETE', headers=HEADERS, **self.creds)
            except Exception as error:
                self.module.fail_json(msg='Failed to update storage array lun mapping. Id [%s]. Error [%s]' % (self.ssid, to_native(error)))
        self.module.exit_json(msg='Lun mapping is complete.', changed=update)