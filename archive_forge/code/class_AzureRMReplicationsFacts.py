from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMReplicationsFacts(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), registry_name=dict(type='str', required=True), replication_name=dict(type='str', required=True))
        self.results = dict(changed=False, ansible_facts=dict())
        self.resource_group = None
        self.registry_name = None
        self.replication_name = None
        super(AzureRMReplicationsFacts, self).__init__(self.module_arg_spec, supports_tags=False, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.registry_name is not None and (self.replication_name is not None):
            self.results['replications'] = self.get()
        return self.results

    def get(self):
        """
        Gets facts of the specified Replication.

        :return: deserialized Replicationinstance state dictionary
        """
        response = None
        results = {}
        try:
            response = self.containerregistry_client.replications.get(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Replications: {0}'.format(str(e)))
        if response is not None:
            results[response.name] = response.as_dict()
        return results