from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
class DockerSwarmManager(DockerBaseClass):

    def __init__(self, client, results):
        super(DockerSwarmManager, self).__init__()
        self.client = client
        self.results = results
        self.verbose_output = self.client.module.params['verbose_output']
        listed_objects = ['tasks', 'services', 'nodes']
        self.client.fail_task_if_not_swarm_manager()
        self.results['swarm_facts'] = self.get_docker_swarm_facts()
        for docker_object in listed_objects:
            if self.client.module.params[docker_object]:
                returned_name = docker_object
                filter_name = docker_object + '_filters'
                filters = clean_dict_booleans_for_docker_api(client.module.params.get(filter_name))
                self.results[returned_name] = self.get_docker_items_list(docker_object, filters)
        if self.client.module.params['unlock_key']:
            self.results['swarm_unlock_key'] = self.get_docker_swarm_unlock_key()

    def get_docker_swarm_facts(self):
        try:
            return self.client.inspect_swarm()
        except APIError as exc:
            self.client.fail('Error inspecting docker swarm: %s' % to_native(exc))

    def get_docker_items_list(self, docker_object=None, filters=None):
        items = None
        items_list = []
        try:
            if docker_object == 'nodes':
                items = self.client.nodes(filters=filters)
            elif docker_object == 'tasks':
                items = self.client.tasks(filters=filters)
            elif docker_object == 'services':
                items = self.client.services(filters=filters)
        except APIError as exc:
            self.client.fail("Error inspecting docker swarm for object '%s': %s" % (docker_object, to_native(exc)))
        if self.verbose_output:
            return items
        for item in items:
            item_record = dict()
            if docker_object == 'nodes':
                item_record = self.get_essential_facts_nodes(item)
            elif docker_object == 'tasks':
                item_record = self.get_essential_facts_tasks(item)
            elif docker_object == 'services':
                item_record = self.get_essential_facts_services(item)
                if item_record['Mode'] == 'Global':
                    item_record['Replicas'] = len(items)
            items_list.append(item_record)
        return items_list

    @staticmethod
    def get_essential_facts_nodes(item):
        object_essentials = dict()
        object_essentials['ID'] = item.get('ID')
        object_essentials['Hostname'] = item['Description']['Hostname']
        object_essentials['Status'] = item['Status']['State']
        object_essentials['Availability'] = item['Spec']['Availability']
        if 'ManagerStatus' in item:
            object_essentials['ManagerStatus'] = item['ManagerStatus']['Reachability']
            if 'Leader' in item['ManagerStatus'] and item['ManagerStatus']['Leader'] is True:
                object_essentials['ManagerStatus'] = 'Leader'
        else:
            object_essentials['ManagerStatus'] = None
        object_essentials['EngineVersion'] = item['Description']['Engine']['EngineVersion']
        return object_essentials

    def get_essential_facts_tasks(self, item):
        object_essentials = dict()
        object_essentials['ID'] = item['ID']
        object_essentials['ContainerID'] = item['Status']['ContainerStatus']['ContainerID']
        object_essentials['Image'] = item['Spec']['ContainerSpec']['Image']
        object_essentials['Node'] = self.client.get_node_name_by_id(item['NodeID'])
        object_essentials['DesiredState'] = item['DesiredState']
        object_essentials['CurrentState'] = item['Status']['State']
        if 'Err' in item['Status']:
            object_essentials['Error'] = item['Status']['Err']
        else:
            object_essentials['Error'] = None
        return object_essentials

    @staticmethod
    def get_essential_facts_services(item):
        object_essentials = dict()
        object_essentials['ID'] = item['ID']
        object_essentials['Name'] = item['Spec']['Name']
        if 'Replicated' in item['Spec']['Mode']:
            object_essentials['Mode'] = 'Replicated'
            object_essentials['Replicas'] = item['Spec']['Mode']['Replicated']['Replicas']
        elif 'Global' in item['Spec']['Mode']:
            object_essentials['Mode'] = 'Global'
            object_essentials['Replicas'] = None
        object_essentials['Image'] = item['Spec']['TaskTemplate']['ContainerSpec']['Image']
        if item['Spec'].get('EndpointSpec') and 'Ports' in item['Spec']['EndpointSpec']:
            object_essentials['Ports'] = item['Spec']['EndpointSpec']['Ports']
        else:
            object_essentials['Ports'] = []
        return object_essentials

    def get_docker_swarm_unlock_key(self):
        unlock_key = self.client.get_unlock_key() or {}
        return unlock_key.get('UnlockKey') or None