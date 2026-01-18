from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
class AnsibleHCloudFirewall(AnsibleHCloud):
    represent = 'hcloud_firewall'
    hcloud_firewall: BoundFirewall | None = None

    def _prepare_result(self):
        return {'id': to_native(self.hcloud_firewall.id), 'name': to_native(self.hcloud_firewall.name), 'rules': [self._prepare_result_rule(rule) for rule in self.hcloud_firewall.rules], 'labels': self.hcloud_firewall.labels, 'applied_to': [self._prepare_result_applied_to(resource) for resource in self.hcloud_firewall.applied_to]}

    def _prepare_result_rule(self, rule: FirewallRule):
        return {'direction': to_native(rule.direction), 'protocol': to_native(rule.protocol), 'port': to_native(rule.port) if rule.port is not None else None, 'source_ips': [to_native(cidr) for cidr in rule.source_ips], 'destination_ips': [to_native(cidr) for cidr in rule.destination_ips], 'description': to_native(rule.description) if rule.description is not None else None}

    def _prepare_result_applied_to(self, resource: FirewallResource):
        result = {'type': to_native(resource.type), 'server': to_native(resource.server.id) if resource.server is not None else None, 'label_selector': to_native(resource.label_selector.selector) if resource.label_selector is not None else None}
        if resource.applied_to_resources is not None:
            result['applied_to_resources'] = [{'type': to_native(item.type), 'server': to_native(item.server.id) if item.server is not None else None} for item in resource.applied_to_resources]
        return result

    def _get_firewall(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_firewall = self.client.firewalls.get_by_id(self.module.params.get('id'))
            elif self.module.params.get('name') is not None:
                self.hcloud_firewall = self.client.firewalls.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_firewall(self):
        self.module.fail_on_missing_params(required_params=['name'])
        params = {'name': self.module.params.get('name'), 'labels': self.module.params.get('labels')}
        rules = self.module.params.get('rules')
        if rules is not None:
            params['rules'] = [FirewallRule(direction=rule['direction'], protocol=rule['protocol'], source_ips=rule['source_ips'] if rule['source_ips'] is not None else [], destination_ips=rule['destination_ips'] if rule['destination_ips'] is not None else [], port=rule['port'], description=rule['description']) for rule in rules]
        if not self.module.check_mode:
            try:
                self.client.firewalls.create(**params)
            except HCloudException as exception:
                self.fail_json_hcloud(exception, params=params)
        self._mark_as_changed()
        self._get_firewall()

    def _update_firewall(self):
        name = self.module.params.get('name')
        if name is not None and self.hcloud_firewall.name != name:
            self.module.fail_on_missing_params(required_params=['id'])
            if not self.module.check_mode:
                self.hcloud_firewall.update(name=name)
            self._mark_as_changed()
        labels = self.module.params.get('labels')
        if labels is not None and self.hcloud_firewall.labels != labels:
            if not self.module.check_mode:
                self.hcloud_firewall.update(labels=labels)
            self._mark_as_changed()
        rules = self.module.params.get('rules')
        if rules is not None and rules != [self._prepare_result_rule(rule) for rule in self.hcloud_firewall.rules]:
            if not self.module.check_mode:
                new_rules = [FirewallRule(direction=rule['direction'], protocol=rule['protocol'], source_ips=rule['source_ips'] if rule['source_ips'] is not None else [], destination_ips=rule['destination_ips'] if rule['destination_ips'] is not None else [], port=rule['port'], description=rule['description']) for rule in rules]
                self.hcloud_firewall.set_rules(new_rules)
            self._mark_as_changed()
        self._get_firewall()

    def present_firewall(self):
        self._get_firewall()
        if self.hcloud_firewall is None:
            self._create_firewall()
        else:
            self._update_firewall()

    def delete_firewall(self):
        self._get_firewall()
        if self.hcloud_firewall is not None:
            if not self.module.check_mode:
                if self.hcloud_firewall.applied_to:
                    if self.module.params.get('force'):
                        actions = self.hcloud_firewall.remove_from_resources(self.hcloud_firewall.applied_to)
                        for action in actions:
                            action.wait_until_finished()
                    else:
                        self.module.warn(f'Firewall {self.hcloud_firewall.name} is currently used by other resources. You need to unassign the resources before deleting the Firewall or use force=true.')
                retry_count = 0
                while True:
                    try:
                        self.hcloud_firewall.delete()
                        break
                    except APIException as exception:
                        if 'is still in use' in exception.message and retry_count < 10:
                            retry_count += 1
                            time.sleep(0.5 * retry_count)
                            continue
                        self.fail_json_hcloud(exception)
                    except HCloudException as exception:
                        self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_firewall = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, labels={'type': 'dict'}, rules=dict(type='list', elements='dict', options=dict(description={'type': 'str'}, direction={'type': 'str', 'choices': ['in', 'out']}, protocol={'type': 'str', 'choices': ['icmp', 'udp', 'tcp', 'esp', 'gre']}, port={'type': 'str'}, source_ips={'type': 'list', 'elements': 'str', 'default': []}, destination_ips={'type': 'list', 'elements': 'str', 'default': []}), required_together=[['direction', 'protocol']], required_if=[['protocol', 'udp', ['port']], ['protocol', 'tcp', ['port']]]), force={'type': 'bool', 'default': False}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], required_if=[['state', 'present', ['name']]], supports_check_mode=True)