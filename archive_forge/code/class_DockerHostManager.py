from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, APIError
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import convert_filters
class DockerHostManager(DockerBaseClass):

    def __init__(self, client, results):
        super(DockerHostManager, self).__init__()
        self.client = client
        self.results = results
        self.verbose_output = self.client.module.params['verbose_output']
        listed_objects = ['volumes', 'networks', 'containers', 'images']
        self.results['host_info'] = self.get_docker_host_info()
        if self.client.module.params['disk_usage']:
            self.results['disk_usage'] = self.get_docker_disk_usage_facts()
        for docker_object in listed_objects:
            if self.client.module.params[docker_object]:
                returned_name = docker_object
                filter_name = docker_object + '_filters'
                filters = clean_dict_booleans_for_docker_api(client.module.params.get(filter_name), True)
                self.results[returned_name] = self.get_docker_items_list(docker_object, filters)

    def get_docker_host_info(self):
        try:
            return self.client.info()
        except APIError as exc:
            self.client.fail('Error inspecting docker host: %s' % to_native(exc))

    def get_docker_disk_usage_facts(self):
        try:
            if self.verbose_output:
                return self.client.df()
            else:
                return dict(LayersSize=self.client.df()['LayersSize'])
        except APIError as exc:
            self.client.fail('Error inspecting docker host: %s' % to_native(exc))

    def get_docker_items_list(self, docker_object=None, filters=None, verbose=False):
        items = None
        items_list = []
        header_containers = ['Id', 'Image', 'Command', 'Created', 'Status', 'Ports', 'Names']
        header_volumes = ['Driver', 'Name']
        header_images = ['Id', 'RepoTags', 'Created', 'Size']
        header_networks = ['Id', 'Driver', 'Name', 'Scope']
        filter_arg = dict()
        if filters:
            filter_arg['filters'] = filters
        try:
            if docker_object == 'containers':
                params = {'limit': -1, 'all': 1 if self.client.module.params['containers_all'] else 0, 'size': 0, 'trunc_cmd': 0, 'filters': convert_filters(filters) if filters else None}
                items = self.client.get_json('/containers/json', params=params)
            elif docker_object == 'networks':
                params = {'filters': convert_filters(filters or {})}
                items = self.client.get_json('/networks', params=params)
            elif docker_object == 'images':
                params = {'only_ids': 0, 'all': 0, 'filters': convert_filters(filters) if filters else None}
                items = self.client.get_json('/images/json', params=params)
            elif docker_object == 'volumes':
                params = {'filters': convert_filters(filters) if filters else None}
                items = self.client.get_json('/volumes', params=params)
                items = items['Volumes']
        except APIError as exc:
            self.client.fail("Error inspecting docker host for object '%s': %s" % (docker_object, to_native(exc)))
        if self.verbose_output:
            return items
        for item in items:
            item_record = dict()
            if docker_object == 'containers':
                for key in header_containers:
                    item_record[key] = item.get(key)
            elif docker_object == 'networks':
                for key in header_networks:
                    item_record[key] = item.get(key)
            elif docker_object == 'images':
                for key in header_images:
                    item_record[key] = item.get(key)
            elif docker_object == 'volumes':
                for key in header_volumes:
                    item_record[key] = item.get(key)
            items_list.append(item_record)
        return items_list