from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOKubernetes(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.return_kubeconfig = self.module.params.pop('return_kubeconfig', False)
        self.wait = self.module.params.pop('wait', True)
        self.wait_timeout = self.module.params.pop('wait_timeout', 600)
        self.module.params.pop('oauth_token')
        self.cluster_id = None
        if self.module.params.get('project_name'):
            self.projects = DigitalOceanProjects(module, self.rest)

    def get_by_id(self):
        """Returns an existing DigitalOcean Kubernetes cluster matching on id"""
        response = self.rest.get('kubernetes/clusters/{0}'.format(self.cluster_id))
        json_data = response.json
        if response.status_code == 200:
            return json_data
        return None

    def get_all_clusters(self):
        """Returns all DigitalOcean Kubernetes clusters"""
        response = self.rest.get('kubernetes/clusters')
        json_data = response.json
        if response.status_code == 200:
            return json_data
        return None

    def get_by_name(self, cluster_name):
        """Returns an existing DigitalOcean Kubernetes cluster matching on name"""
        if not cluster_name:
            return None
        clusters = self.get_all_clusters()
        for cluster in clusters['kubernetes_clusters']:
            if cluster['name'] == cluster_name:
                return cluster
        return None

    def get_kubernetes_kubeconfig(self):
        """Returns the kubeconfig for an existing DigitalOcean Kubernetes cluster"""
        response = self.rest.get('kubernetes/clusters/{0}/kubeconfig'.format(self.cluster_id))
        if response.status_code == 200:
            return response.body
        else:
            self.module.fail_json(msg='Failed to retrieve kubeconfig')

    def get_kubernetes(self):
        """Returns an existing DigitalOcean Kubernetes cluster by name"""
        json_data = self.get_by_name(self.module.params['name'])
        if json_data:
            self.cluster_id = json_data['id']
            return json_data
        else:
            return None

    def get_kubernetes_options(self):
        """Fetches DigitalOcean Kubernetes options: regions, sizes, versions.
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/list_kubernetes_options
        """
        response = self.rest.get('kubernetes/options')
        json_data = response.json
        if response.status_code == 200:
            return json_data
        return None

    def ensure_running(self):
        """Waits for the newly created DigitalOcean Kubernetes cluster to be running"""
        end_time = time.monotonic() + self.wait_timeout
        while time.monotonic() < end_time:
            cluster = self.get_by_id()
            if cluster['kubernetes_cluster']['status']['state'] == 'running':
                return cluster
            time.sleep(10)
        self.module.fail_json(msg='Wait for Kubernetes cluster to be running')

    def create(self):
        """Creates a DigitalOcean Kubernetes cluster
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/create_kubernetes_cluster
        """
        kubernetes_options = self.get_kubernetes_options()['options']
        valid_regions = [str(x['slug']) for x in kubernetes_options['regions']]
        if self.module.params.get('region') not in valid_regions:
            self.module.fail_json(msg='Invalid region {0} (valid regions are {1})'.format(self.module.params.get('region'), ', '.join(valid_regions)))
        valid_versions = [str(x['slug']) for x in kubernetes_options['versions']]
        valid_versions.append('latest')
        if self.module.params.get('version') not in valid_versions:
            self.module.fail_json(msg='Invalid version {0} (valid versions are {1})'.format(self.module.params.get('version'), ', '.join(valid_versions)))
        valid_sizes = [str(x['slug']) for x in kubernetes_options['sizes']]
        for node_pool in self.module.params.get('node_pools'):
            if node_pool['size'] not in valid_sizes:
                self.module.fail_json(msg='Invalid size {0} (valid sizes are {1})'.format(node_pool['size'], ', '.join(valid_sizes)))
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        json_data = self.get_kubernetes()
        if json_data:
            if self.return_kubeconfig:
                json_data['kubeconfig'] = self.get_kubernetes_kubeconfig()
            project_name = self.module.params.get('project_name')
            if project_name:
                urn = 'do:kubernetes:{0}'.format(self.cluster_id)
                assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
                if assign_status not in {'ok', 'assigned', 'already_assigned'}:
                    self.module.fail_json(changed=False, msg=error_message, assign_status=assign_status, resources=resources)
            self.module.exit_json(changed=False, data=json_data)
        request_params = dict(self.module.params)
        response = self.rest.post('kubernetes/clusters', data=request_params)
        json_data = response.json
        if response.status_code >= 400:
            self.module.fail_json(changed=False, msg=json_data)
        self.cluster_id = json_data['kubernetes_cluster']['id']
        if self.wait:
            json_data = self.ensure_running()
        if self.return_kubeconfig:
            json_data['kubeconfig'] = self.get_kubernetes_kubeconfig()
        project_name = self.module.params.get('project_name')
        if project_name:
            urn = 'do:kubernetes:{0}'.format(self.cluster_id)
            assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
            if assign_status not in {'ok', 'assigned', 'already_assigned'}:
                self.module.fail_json(changed=True, msg=error_message, assign_status=assign_status, resources=resources)
            json_data['kubernetes_cluster']['kubeconfig'] = self.get_kubernetes_kubeconfig()
        self.module.exit_json(changed=True, data=json_data['kubernetes_cluster'])

    def delete(self):
        """Deletes a DigitalOcean Kubernetes cluster
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/delete_kubernetes_cluster
        """
        json_data = self.get_kubernetes()
        if json_data:
            if self.module.check_mode:
                self.module.exit_json(changed=True)
            response = self.rest.delete('kubernetes/clusters/{0}'.format(json_data['id']))
            if response.status_code == 204:
                self.module.exit_json(changed=True, data=json_data, msg='Kubernetes cluster deleted')
            self.module.fail_json(changed=False, msg='Failed to delete Kubernetes cluster')
            json_data = response.json
        else:
            self.module.exit_json(changed=False, msg='Kubernetes cluster not found')