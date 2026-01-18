from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMIoTHubConsumerGroup(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), hub=dict(type='str', required=True), event_hub=dict(type='str', default='events'))
        self.results = dict(changed=False, id=None)
        self.resource_group = None
        self.name = None
        self.state = None
        self.hub = None
        self.event_hub = None
        super(AzureRMIoTHubConsumerGroup, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec.keys():
            setattr(self, key, kwargs[key])
        changed = False
        cg = self.get_cg()
        if not cg and self.state == 'present':
            changed = True
            if not self.check_mode:
                cg = self.create_cg()
        elif cg and self.state == 'absent':
            changed = True
            cg = None
            if not self.check_mode:
                self.delete_cg()
        self.results = dict(id=cg.id, name=cg.name) if cg else dict()
        self.results['changed'] = changed
        return self.results

    def get_cg(self):
        try:
            return self.IoThub_client.iot_hub_resource.get_event_hub_consumer_group(self.resource_group, self.hub, self.event_hub, self.name)
        except Exception:
            pass
            return None

    def create_cg(self):
        try:
            return self.IoThub_client.iot_hub_resource.create_event_hub_consumer_group(self.resource_group, self.hub, self.event_hub, self.name)
        except Exception as exc:
            self.fail('Error when creating the consumer group {0} for IoT Hub {1} event hub {2}: {3}'.format(self.name, self.hub, self.event_hub, str(exc)))

    def delete_cg(self):
        try:
            return self.IoThub_client.iot_hub_resource.delete_event_hub_consumer_group(self.resource_group, self.hub, self.event_hub, self.name)
        except Exception as exc:
            self.fail('Error when deleting the consumer group {0} for IoT Hub {1} event hub {2}: {3}'.format(self.name, self.hub, self.event_hub, str(exc)))