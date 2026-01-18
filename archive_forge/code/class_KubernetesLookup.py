from __future__ import absolute_import, division, print_function
import os
from ansible.errors import AnsibleError
from ansible.module_utils.common._collections_compat import KeysView
from ansible.module_utils.common.validation import check_type_bool
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.resource import (
class KubernetesLookup(object):

    def __init__(self):
        if not HAS_K8S_MODULE_HELPER:
            raise Exception('Requires the Kubernetes Python client. Try `pip install kubernetes`. Detail: {0}'.format(k8s_import_exception))
        self.kind = None
        self.name = None
        self.namespace = None
        self.api_version = None
        self.label_selector = None
        self.field_selector = None
        self.include_uninitialized = None
        self.resource_definition = None
        self.helper = None
        self.connection = {}

    def fail(self, msg=None):
        raise AnsibleError(msg)

    def run(self, terms, variables=None, **kwargs):
        self.params = kwargs
        self.client = get_api_client(**kwargs)
        cluster_info = kwargs.get('cluster_info')
        if cluster_info == 'version':
            return [self.client.client.version]
        if cluster_info == 'api_groups':
            if isinstance(self.client.resources.api_groups, KeysView):
                return [list(self.client.resources.api_groups)]
            return [self.client.resources.api_groups]
        self.kind = kwargs.get('kind')
        self.name = kwargs.get('resource_name')
        self.namespace = kwargs.get('namespace')
        self.api_version = kwargs.get('api_version', 'v1')
        self.label_selector = kwargs.get('label_selector')
        self.field_selector = kwargs.get('field_selector')
        self.include_uninitialized = kwargs.get('include_uninitialized', False)
        resource_definition = kwargs.get('resource_definition')
        src = kwargs.get('src')
        if src:
            definitions = create_definitions(params=dict(src=src))
            if definitions:
                self.kind = definitions[0].kind
                self.name = definitions[0].name
                self.namespace = definitions[0].namespace
                self.api_version = definitions[0].api_version or 'v1'
        if resource_definition:
            self.kind = resource_definition.get('kind', self.kind)
            self.api_version = resource_definition.get('apiVersion', self.api_version)
            self.name = resource_definition.get('metadata', {}).get('name', self.name)
            self.namespace = resource_definition.get('metadata', {}).get('namespace', self.namespace)
        if not self.kind:
            raise AnsibleError("Error: no Kind specified. Use the 'kind' parameter, or provide an object YAML configuration using the 'resource_definition' parameter.")
        resource = self.client.resource(self.kind, self.api_version)
        try:
            params = dict(name=self.name, namespace=self.namespace, label_selector=self.label_selector, field_selector=self.field_selector)
            k8s_obj = self.client.get(resource, **params)
        except NotFoundError:
            return []
        if self.name:
            return [k8s_obj.to_dict()]
        return k8s_obj.to_dict().get('items')