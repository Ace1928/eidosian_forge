from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOKubernetesInfo(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.module.params.pop('oauth_token')
        self.return_kubeconfig = self.module.params.pop('return_kubeconfig')
        self.cluster_id = None

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

    def get(self):
        """Fetches an existing DigitalOcean Kubernetes cluster
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/list_all_kubernetes_clusters
        """
        json_data = self.get_kubernetes()
        if json_data:
            if self.return_kubeconfig:
                json_data['kubeconfig'] = self.get_kubernetes_kubeconfig()
            self.module.exit_json(changed=False, data=json_data)
        self.module.fail_json(changed=False, msg='Kubernetes cluster not found')