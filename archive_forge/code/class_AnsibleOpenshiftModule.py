from __future__ import (absolute_import, division, print_function)
import traceback
from abc import abstractmethod
from ansible.module_utils._text import to_native
class AnsibleOpenshiftModule(AnsibleK8SModule):

    def __init__(self, **kwargs):
        super(AnsibleOpenshiftModule, self).__init__(**kwargs)
        self.client = get_api_client(module=self)
        self.fail = self.fail_json
        self.svc = K8sService(self.client, self._module)
        self.find_resource = self.svc.find_resource
        self.kubernetes_facts = self.svc.find
        if not HAS_KUBERNETES_COLLECTION:
            self.fail_json(msg='The kubernetes.core collection must be installed', exception=K8S_COLLECTION_ERROR, error=to_native(k8s_collection_import_exception))

    @property
    def module(self):
        return self._module

    @abstractmethod
    def execute_module(self):
        pass

    def request(self, *args, **kwargs):
        return self.client.client.request(*args, **kwargs)

    def set_resource_definitions(self):
        self.resource_definitions = create_definitions(self.params)

    def perform_action(self, definition, params):
        return perform_action(self.svc, definition, params)

    def validate(self, definition):
        validate(self.client, self, definition)

    @staticmethod
    def merge_params(definition, params):
        return merge_params(definition, params)

    @staticmethod
    def flatten_list_kind(definition, params):
        return flatten_list_kind(definition, params)

    @staticmethod
    def diff_objects(existing, new):
        return diff_objects(existing, new)

    def run_module(self):
        try:
            self.execute_module()
        except CoreException as e:
            self.fail_from_exception(e)