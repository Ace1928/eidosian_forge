from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.rabbitmq.plugins.module_utils.rabbitmq import rabbitmq_argument_spec
class RabbitMqBinding(object):

    def __init__(self, module):
        """
        :param module:
        """
        self.module = module
        self.name = self.module.params['name']
        self.login_user = self.module.params['login_user']
        self.login_password = self.module.params['login_password']
        self.login_host = self.module.params['login_host']
        self.login_port = self.module.params['login_port']
        self.login_protocol = self.module.params['login_protocol']
        self.vhost = self.module.params['vhost']
        self.destination = self.module.params['destination']
        self.destination_type = 'q' if self.module.params['destination_type'] == 'queue' else 'e'
        self.routing_key = self.module.params['routing_key']
        self.arguments = self.module.params['arguments']
        self.verify = self.module.params['ca_cert']
        self.cert = self.module.params['client_cert']
        self.key = self.module.params['client_key']
        self.props = urllib_parse.quote(self.routing_key) if self.routing_key != '' else '~'
        self.base_url = '{0}://{1}:{2}/api/bindings'.format(self.login_protocol, self.login_host, self.login_port)
        self.url = '{0}/{1}/e/{2}/{3}/{4}/{5}'.format(self.base_url, urllib_parse.quote(self.vhost, safe=''), urllib_parse.quote(self.name, safe=''), self.destination_type, urllib_parse.quote(self.destination, safe=''), self.props)
        self.result = {'changed': False, 'name': self.module.params['name']}
        self.authentication = (self.login_user, self.login_password)
        self.request = requests
        self.http_check_states = {200: True, 404: False}
        self.http_actionable_states = {201: True, 204: True}
        self.api_result = self.request.get(self.url, auth=self.authentication, verify=self.verify, cert=(self.cert, self.key))

    def run(self):
        """
        :return:
        """
        self.check_presence()
        self.check_mode()
        self.action_mode()

    def check_presence(self):
        """
        :return:
        """
        if self.check_should_throw_fail():
            self.fail()

    def change_required(self):
        """
        :return:
        """
        if self.module.params['state'] == 'present':
            if not self.is_present():
                return True
        elif self.module.params['state'] == 'absent':
            if self.is_present():
                return True
        return False

    def is_present(self):
        """
        :return:
        """
        return self.http_check_states.get(self.api_result.status_code, False)

    def check_mode(self):
        """
        :return:
        """
        if self.module.check_mode:
            result = self.result
            result['changed'] = self.change_required()
            result['details'] = self.api_result.json() if self.is_present() else self.api_result.text
            result['arguments'] = self.module.params['arguments']
            self.module.exit_json(**result)

    def check_reply_is_correct(self):
        """
        :return:
        """
        if self.api_result.status_code in self.http_check_states:
            return True
        return False

    def check_should_throw_fail(self):
        """
        :return:
        """
        if not self.is_present():
            if not self.check_reply_is_correct():
                return True
        return False

    def action_mode(self):
        """
        :return:
        """
        result = self.result
        if self.change_required():
            if self.module.params['state'] == 'present':
                self.create()
            if self.module.params['state'] == 'absent':
                self.remove()
            if self.action_should_throw_fail():
                self.fail()
            result['changed'] = True
            result['destination'] = self.module.params['destination']
            self.module.exit_json(**result)
        else:
            result['changed'] = False
            self.module.exit_json(**result)

    def action_reply_is_correct(self):
        """
        :return:
        """
        if self.api_result.status_code in self.http_actionable_states:
            return True
        return False

    def action_should_throw_fail(self):
        """
        :return:
        """
        if not self.action_reply_is_correct():
            return True
        return False

    def create(self):
        """
        :return:
        """
        self.url = '{0}/{1}/e/{2}/{3}/{4}'.format(self.base_url, urllib_parse.quote(self.vhost, safe=''), urllib_parse.quote(self.name, safe=''), self.destination_type, urllib_parse.quote(self.destination, safe=''))
        self.api_result = self.request.post(self.url, auth=self.authentication, verify=self.verify, cert=(self.cert, self.key), headers={'content-type': 'application/json'}, data=json.dumps({'routing_key': self.routing_key, 'arguments': self.arguments}))

    def remove(self):
        """
        :return:
        """
        self.api_result = self.request.delete(self.url, auth=self.authentication, verify=self.verify, cert=(self.cert, self.key))

    def fail(self):
        """
        :return:
        """
        self.module.fail_json(msg='Unexpected reply from API', status=self.api_result.status_code, details=self.api_result.text)