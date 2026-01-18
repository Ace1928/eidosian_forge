from __future__ import absolute_import, division, print_function
class AzureRMServiceBusSASPolicy(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), namespace=dict(type='str', required=True), queue=dict(type='str'), topic=dict(type='str'), regenerate_primary_key=dict(type='bool', default=False), regenerate_secondary_key=dict(type='bool', default=False), rights=dict(type='str', choices=['manage', 'listen', 'send', 'listen_send']))
        mutually_exclusive = [['queue', 'topic']]
        required_if = [('state', 'present', ['rights'])]
        self.resource_group = None
        self.name = None
        self.state = None
        self.namespace = None
        self.queue = None
        self.topic = None
        self.regenerate_primary_key = False
        self.regenerate_secondary_key = False
        self.rights = None
        self.results = dict(changed=False, id=None)
        super(AzureRMServiceBusSASPolicy, self).__init__(self.module_arg_spec, mutually_exclusive=mutually_exclusive, required_if=required_if, supports_tags=False, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        changed = False
        policy = self.get_auth_rule()
        if self.state == 'present':
            if not policy:
                changed = True
                if not self.check_mode:
                    policy = self.create_sas_policy()
            else:
                changed = changed | self.regenerate_primary_key | self.regenerate_secondary_key
                if self.regenerate_primary_key and (not self.check_mode):
                    self.regenerate_sas_key('primary')
                if self.regenerate_secondary_key and (not self.check_mode):
                    self.regenerate_sas_key('secondary')
            self.results = self.policy_to_dict(policy)
            self.results['keys'] = self.get_sas_key()
        elif policy:
            changed = True
            if not self.check_mode:
                self.delete_sas_policy()
        self.results['changed'] = changed
        return self.results

    def _get_client(self):
        if self.queue:
            return self.servicebus_client.queues
        elif self.topic:
            return self.servicebus_client.topics
        return self.servicebus_client.namespaces

    def create_sas_policy(self):
        if self.rights == 'listen_send':
            rights = ['Listen', 'Send']
        elif self.rights == 'manage':
            rights = ['Listen', 'Send', 'Manage']
        else:
            rights = [str.capitalize(self.rights)]
        try:
            client = self._get_client()
            if self.queue or self.topic:
                rule = client.create_or_update_authorization_rule(self.resource_group, self.namespace, self.queue or self.topic, self.name, parameters={'rights': rights})
            else:
                rule = client.create_or_update_authorization_rule(self.resource_group, self.namespace, self.name, parameters={'rights': rights})
            return rule
        except Exception as exc:
            self.fail('Error when creating or updating SAS policy {0} - {1}'.format(self.name, exc.message or str(exc)))
        return None

    def get_auth_rule(self):
        rule = None
        try:
            client = self._get_client()
            if self.queue or self.topic:
                rule = client.get_authorization_rule(self.resource_group, self.namespace, self.queue or self.topic, self.name)
            else:
                rule = client.get_authorization_rule(self.resource_group, self.namespace, self.name)
        except Exception:
            pass
        return rule

    def delete_sas_policy(self):
        try:
            client = self._get_client()
            if self.queue or self.topic:
                client.delete_authorization_rule(self.resource_group, self.namespace, self.queue or self.topic, self.name)
            else:
                client.delete_authorization_rule(self.resource_group, self.namespace, self.name)
            return True
        except Exception as exc:
            self.fail('Error when deleting SAS policy {0} - {1}'.format(self.name, exc.message or str(exc)))

    def regenerate_sas_key(self, key_type):
        try:
            client = self._get_client()
            key = str.capitalize(key_type) + 'Key'
            if self.queue or self.topic:
                client.regenerate_keys(self.resource_group, self.namespace, self.queue or self.topic, self.name, key)
            else:
                client.regenerate_keys(self.resource_group, self.namespace, self.name, key)
        except Exception as exc:
            self.fail("Error when generating SAS policy {0}'s key - {1}".format(self.name, exc.message or str(exc)))
        return None

    def get_sas_key(self):
        try:
            client = self._get_client()
            if self.queue or self.topic:
                return client.list_keys(self.resource_group, self.namespace, self.queue or self.topic, self.name).as_dict()
            else:
                return client.list_keys(self.resource_group, self.namespace, self.name).as_dict()
        except Exception:
            pass
        return None

    def policy_to_dict(self, rule):
        result = rule.as_dict()
        rights = result['rights']
        if 'Manage' in rights:
            result['rights'] = 'manage'
        elif 'Listen' in rights and 'Send' in rights:
            result['rights'] = 'listen_send'
        else:
            result['rights'] = rights[0].lower()
        return result