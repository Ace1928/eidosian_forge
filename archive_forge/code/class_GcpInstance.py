from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
class GcpInstance(object):

    def __init__(self, json, hostname_ordering, project_disks, should_format=True, name_suffix=''):
        self.hostname_ordering = hostname_ordering
        self.project_disks = project_disks
        self.name_suffix = name_suffix
        self.json = json
        if should_format:
            self.convert()

    def to_json(self):
        return self.json

    def convert(self):
        if 'zone' in self.json:
            self.json['zone_selflink'] = self.json['zone']
            self.json['zone'] = self.json['zone'].split('/')[-1]
        if 'machineType' in self.json:
            self.json['machineType_selflink'] = self.json['machineType']
            self.json['machineType'] = self.json['machineType'].split('/')[-1]
        if 'networkInterfaces' in self.json:
            for network in self.json['networkInterfaces']:
                if 'network' in network:
                    network['network'] = self._format_network_info(network['network'])
                if 'subnetwork' in network:
                    network['subnetwork'] = self._format_network_info(network['subnetwork'])
        if 'metadata' in self.json:
            self.json['metadata'] = self._format_metadata(self.json['metadata'].get('items', {}))
        self.json['project'] = self.json['selfLink'].split('/')[6]
        self.json['image'] = self._get_image()

    def _format_network_info(self, address):
        """
        :param address: A GCP network address
        :return a dict with network shortname and region
        """
        split = address.split('/')
        region = ''
        if 'global' in split:
            region = 'global'
        else:
            region = split[8]
        return {'region': region, 'name': split[-1], 'selfLink': address}

    def _format_metadata(self, metadata):
        """
        :param metadata: A list of dicts where each dict has keys "key" and "value"
        :return a dict with key/value pairs for each in list.
        """
        new_metadata = {}
        for pair in metadata:
            new_metadata[pair['key']] = pair['value']
        return new_metadata

    def hostname(self):
        """
        :return the hostname of this instance
        """
        for order in self.hostname_ordering:
            name = None
            if order.startswith('labels.'):
                if 'labels' in self.json:
                    name = self.json['labels'].get(order[7:])
            elif order == 'public_ip':
                name = self._get_publicip()
            elif order == 'private_ip':
                name = self._get_privateip()
            elif order == 'name':
                name = self.json['name'] + self.name_suffix
            else:
                raise AnsibleParserError('%s is not a valid hostname precedent' % order)
            if name:
                return name
        raise AnsibleParserError('No valid name found for host')

    def _get_publicip(self):
        """
        :return the publicIP of this instance or None
        """
        for interface in self.json['networkInterfaces']:
            if 'accessConfigs' in interface:
                for accessConfig in interface['accessConfigs']:
                    if 'natIP' in accessConfig:
                        return accessConfig['natIP']
        return None

    def _get_image(self):
        """
        :param instance: A instance response from GCP
        :return the image of this instance or None
        """
        image = None
        if self.project_disks and 'disks' in self.json:
            for disk in self.json['disks']:
                if disk.get('boot'):
                    image = self.project_disks[disk['source']]
        return image

    def _get_privateip(self):
        """
        :param item: A host response from GCP
        :return the privateIP of this instance or None
        """
        for interface in self.json['networkInterfaces']:
            if 'networkIP' in interface:
                return interface['networkIP']