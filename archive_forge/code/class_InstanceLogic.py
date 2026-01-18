from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceLogic(object):

    def __init__(self, module):
        self.module = module
        self.current_instances = self.list_instances()
        self.module_instances = []
        instances = self.module.params.get('instances')
        if instances:
            for instance in instances:
                self.module_instances.append(replace_resource_dict(instance, 'selfLink'))

    def run(self):
        instances_to_add = list(set(self.module_instances) - set(self.current_instances))
        if instances_to_add:
            self.add_instances(instances_to_add)
        instances_to_remove = list(set(self.current_instances) - set(self.module_instances))
        if instances_to_remove:
            self.remove_instances(instances_to_remove)

    def list_instances(self):
        auth = GcpSession(self.module, 'compute')
        response = return_if_object(self.module, auth.post(self._list_instances_url(), {'instanceState': 'ALL'}), 'compute#instanceGroupsListInstances')
        instances = []
        for instance in response.get('items', []):
            instances.append(instance['instance'])
        return instances

    def add_instances(self, instances):
        auth = GcpSession(self.module, 'compute')
        wait_for_operation(self.module, auth.post(self._add_instances_url(), self._build_request(instances)))

    def remove_instances(self, instances):
        auth = GcpSession(self.module, 'compute')
        wait_for_operation(self.module, auth.post(self._remove_instances_url(), self._build_request(instances)))

    def _list_instances_url(self):
        return 'https://www.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instanceGroups/{name}/listInstances'.format(**self.module.params)

    def _remove_instances_url(self):
        return 'https://www.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instanceGroups/{name}/removeInstances'.format(**self.module.params)

    def _add_instances_url(self):
        return 'https://www.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instanceGroups/{name}/addInstances'.format(**self.module.params)

    def _build_request(self, instances):
        request = {'instances': []}
        for instance in instances:
            request['instances'].append({'instance': instance})
        return request