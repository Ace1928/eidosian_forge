from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class MediaTypeModule(ZabbixBase):

    def check_if_mediatype_exists(self, name):
        """Checks if mediatype exists.

        Args:
            name: Zabbix mediatype name

        Returns:
            Tuple of (True, `id of the mediatype`) if mediatype exists, (False, None) otherwise
        """
        filter_key_name = 'description'
        filter_key_name = 'name'
        try:
            mediatype_list = self._zapi.mediatype.get({'output': 'extend', 'filter': {filter_key_name: [name]}})
            if len(mediatype_list) < 1:
                return (False, None)
            else:
                return (True, mediatype_list[0]['mediatypeid'])
        except Exception as e:
            self._module.fail_json(msg="Failed to get ID of the mediatype '{name}': {e}".format(name=name, e=e))

    def construct_parameters(self):
        """Translates data to a format suitable for Zabbix API and filters
        the ones that are related to the specified mediatype type.

        Returns:
            A dictionary of arguments that are related to transport type,
            and are in a format that is understandable by Zabbix API.
        """
        truths = {'False': '0', 'True': '1'}
        parameters = dict(status='0' if self._module.params['status'] == 'enabled' else '1', type={'email': '0', 'script': '1', 'sms': '2', 'jabber': '3', 'webhook': '4', 'ez_texting': '100'}.get(self._module.params['type']))
        parameters.update(dict(name=self._module.params['name'], description=self._module.params['description'], maxsessions=str(self._module.params['max_sessions']), maxattempts=str(self._module.params['max_attempts']), attempt_interval=str(self._module.params['attempt_interval'])))
        if self._module.params['message_templates']:
            msg_templates = []
            for template in self._module.params['message_templates']:
                msg_templates.append(dict(eventsource={'triggers': '0', 'discovery': '1', 'autoregistration': '2', 'internal': '3'}.get(template['eventsource']), recovery={'operations': '0', 'recovery_operations': '1', 'update_operations': '2'}.get(template['recovery']), subject=template['subject'], message=template['body']))
            parameters.update(dict(message_templates=msg_templates))
        if self._module.params['type'] == 'email':
            parameters.update(dict(smtp_server=self._module.params['smtp_server'], smtp_port=str(self._module.params['smtp_server_port']), smtp_helo=self._module.params['smtp_helo'], smtp_email=self._module.params['smtp_email'], smtp_security={'None': '0', 'STARTTLS': '1', 'SSL/TLS': '2'}.get(str(self._module.params['smtp_security'])), smtp_authentication=truths.get(str(self._module.params['smtp_authentication'])), smtp_verify_host=truths.get(str(self._module.params['smtp_verify_host'])), smtp_verify_peer=truths.get(str(self._module.params['smtp_verify_peer'])), username=self._module.params['username'], passwd=self._module.params['password']))
            if parameters['smtp_authentication'] == '0':
                parameters.pop('username')
                parameters.pop('passwd')
            return parameters
        elif self._module.params['type'] == 'script':
            if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
                if self._module.params['script_params'] is None:
                    _script_params = ''
                else:
                    _script_params = '\n'.join((str(i) for i in self._module.params['script_params'])) + '\n'
                parameters.update(dict(exec_path=self._module.params['script_name'], exec_params=_script_params))
            else:
                _script_params = []
                if self._module.params['script_params']:
                    for i, val in enumerate(self._module.params['script_params']):
                        _script_params.append({'sortorder': str(i), 'value': val})
                parameters.update(dict(exec_path=self._module.params['script_name'], parameters=_script_params))
            return parameters
        elif self._module.params['type'] == 'sms':
            parameters.update(dict(gsm_modem=self._module.params['gsm_modem']))
            return parameters
        elif self._module.params['type'] == 'webhook':
            parameters.update(dict(script=self._module.params['webhook_script'], timeout=self._module.params['webhook_timeout'], process_tags=truths.get(str(self._module.params['process_tags'])), show_event_menu=truths.get(str(self._module.params['event_menu'])), parameters=self._module.params['webhook_params']))
            if self._module.params['event_menu']:
                parameters.update(dict(event_menu_url=self._module.params['event_menu_url'], event_menu_name=self._module.params['event_menu_name']))
            return parameters
        self._module.fail_json(msg='%s is unsupported for Zabbix version %s' % (self._module.params['type'], self._zbx_api_version))

    def validate_params(self, params):
        """Validates arguments that are required together.

        Fails the module with the message that shows the missing
        requirements if there are some.

        Args:
            params (list): Each element of this list
                is a list like
                ['argument_key', 'argument_value', ['required_arg_1',
                                                    'required_arg_2']].
                Format is the same as `required_if` parameter of AnsibleModule.
        """
        for param in params:
            if self._module.params[param[0]] == param[1]:
                if None in [self._module.params[i] for i in param[2]]:
                    self._module.fail_json(msg='Following arguments are required when {key} is {value}: {arguments}'.format(key=param[0], value=param[1], arguments=', '.join(param[2])))

    def get_update_params(self, mediatype_id, **kwargs):
        """Filters only the parameters that are different and need to be updated.

        Args:
            mediatype_id (int): ID of the mediatype to be updated.
            **kwargs: Parameters for the new mediatype.

        Returns:
            A tuple where the first element is a dictionary of parameters
            that need to be updated and the second one is a dictionary
            returned by diff() function with
            existing mediatype data and new params passed to it.
        """
        get_params = {'output': 'extend', 'mediatypeids': [mediatype_id], 'selectMessageTemplates': 'extend'}
        existing_mediatype = self._zapi.mediatype.get(get_params)[0]
        if existing_mediatype['type'] != kwargs['type']:
            return (kwargs, diff(existing_mediatype, kwargs))
        else:
            params_to_update = {}
            for key in kwargs:
                if key == 'parameters' and (kwargs[key] != [] or existing_mediatype[key] != []):
                    if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
                        kwargs[key] = sorted(kwargs[key], key=lambda x: x['name'])
                        existing_mediatype[key] = sorted(existing_mediatype[key], key=lambda x: x['name'])
                    elif kwargs['type'] == '1':
                        kwargs[key] = sorted(kwargs[key], key=lambda x: x['sortorder'])
                        existing_mediatype[key] = sorted(existing_mediatype[key], key=lambda x: x['sortorder'])
                    elif kwargs['type'] == '4':
                        kwargs[key] = sorted(kwargs[key], key=lambda x: x['name'])
                        existing_mediatype[key] = sorted(existing_mediatype[key], key=lambda x: x['name'])
                if key == 'message_templates' and (kwargs[key] != [] or existing_mediatype[key] != []):
                    kwargs[key] = sorted(kwargs[key], key=lambda x: x['subject'])
                    existing_mediatype[key] = sorted(existing_mediatype[key], key=lambda x: x['subject'])
                if not (kwargs[key] is None and existing_mediatype[key] == '') and kwargs[key] != existing_mediatype[key]:
                    params_to_update[key] = kwargs[key]
            return (params_to_update, diff(existing_mediatype, kwargs))

    def delete_mediatype(self, mediatype_id):
        try:
            return self._zapi.mediatype.delete([mediatype_id])
        except Exception as e:
            self._module.fail_json(msg="Failed to delete mediatype '{_id}': {e}".format(_id=mediatype_id, e=e))

    def update_mediatype(self, **kwargs):
        try:
            self._zapi.mediatype.update(kwargs)
        except Exception as e:
            self._module.fail_json(msg="Failed to update mediatype '{_id}': {e}".format(_id=kwargs['mediatypeid'], e=e))

    def create_mediatype(self, **kwargs):
        try:
            self._zapi.mediatype.create(kwargs)
        except Exception as e:
            self._module.fail_json(msg="Failed to create mediatype '{name}': {e}".format(name=kwargs['name'], e=e))