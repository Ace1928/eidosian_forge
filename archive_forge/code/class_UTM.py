from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class UTM:

    def __init__(self, module, endpoint, change_relevant_keys, info_only=False):
        """
        Initialize UTM Class
        :param module: The Ansible module
        :param endpoint: The corresponding endpoint to the module
        :param change_relevant_keys: The keys of the object to check for changes
        :param info_only: When implementing an info module, set this to true. Will allow access to the info method only
        """
        self.info_only = info_only
        self.module = module
        self.request_url = module.params.get('utm_protocol') + '://' + module.params.get('utm_host') + ':' + to_native(module.params.get('utm_port')) + '/api/objects/' + endpoint + '/'
        '\n        The change_relevant_keys will be checked for changes to determine whether the object needs to be updated\n        '
        self.change_relevant_keys = change_relevant_keys
        self.module.params['url_username'] = 'token'
        self.module.params['url_password'] = module.params.get('utm_token')
        if all((elem in self.change_relevant_keys for elem in module.params.keys())):
            raise UTMModuleConfigurationError('The keys ' + to_native(self.change_relevant_keys) + ' to check are not in the modules keys:\n' + to_native(list(module.params.keys())))

    def execute(self):
        try:
            if not self.info_only:
                if self.module.params.get('state') == 'present':
                    self._add()
                elif self.module.params.get('state') == 'absent':
                    self._remove()
            else:
                self._info()
        except Exception as e:
            self.module.fail_json(msg=to_native(e))

    def _info(self):
        """
        returns the info for an object in utm
        """
        info, result = self._lookup_entry(self.module, self.request_url)
        if info['status'] >= 400:
            self.module.fail_json(result=json.loads(info))
        elif result is None:
            self.module.exit_json(changed=False)
        else:
            self.module.exit_json(result=result, changed=False)

    def _add(self):
        """
        adds or updates a host object on utm
        """
        combined_headers = self._combine_headers()
        is_changed = False
        info, result = self._lookup_entry(self.module, self.request_url)
        if info['status'] >= 400:
            self.module.fail_json(result=json.loads(info))
        else:
            data_as_json_string = self.module.jsonify(self.module.params)
            if result is None:
                response, info = fetch_url(self.module, self.request_url, method='POST', headers=combined_headers, data=data_as_json_string)
                if info['status'] >= 400:
                    self.module.fail_json(msg=json.loads(info['body']))
                is_changed = True
                result = self._clean_result(json.loads(response.read()))
            elif self._is_object_changed(self.change_relevant_keys, self.module, result):
                response, info = fetch_url(self.module, self.request_url + result['_ref'], method='PUT', headers=combined_headers, data=data_as_json_string)
                if info['status'] >= 400:
                    self.module.fail_json(msg=json.loads(info['body']))
                is_changed = True
                result = self._clean_result(json.loads(response.read()))
            self.module.exit_json(result=result, changed=is_changed)

    def _combine_headers(self):
        """
        This will combine a header default with headers that come from the module declaration
        :return: A combined headers dict
        """
        default_headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
        if self.module.params.get('headers') is not None:
            result = default_headers.copy()
            result.update(self.module.params.get('headers'))
        else:
            result = default_headers
        return result

    def _remove(self):
        """
        removes an object from utm
        """
        is_changed = False
        info, result = self._lookup_entry(self.module, self.request_url)
        if result is not None:
            response, info = fetch_url(self.module, self.request_url + result['_ref'], method='DELETE', headers={'Accept': 'application/json', 'X-Restd-Err-Ack': 'all'}, data=self.module.jsonify(self.module.params))
            if info['status'] >= 400:
                self.module.fail_json(msg=json.loads(info['body']))
            else:
                is_changed = True
        self.module.exit_json(changed=is_changed)

    def _lookup_entry(self, module, request_url):
        """
        Lookup for existing entry
        :param module:
        :param request_url:
        :return:
        """
        response, info = fetch_url(module, request_url, method='GET', headers={'Accept': 'application/json'})
        result = None
        if response is not None:
            results = json.loads(response.read())
            result = next(iter(filter(lambda d: d['name'] == module.params.get('name'), results)), None)
        return (info, result)

    def _clean_result(self, result):
        """
        Will clean the result from irrelevant fields
        :param result: The result from the query
        :return: The modified result
        """
        del result['utm_host']
        del result['utm_port']
        del result['utm_token']
        del result['utm_protocol']
        del result['validate_certs']
        del result['url_username']
        del result['url_password']
        del result['state']
        return result

    def _is_object_changed(self, keys, module, result):
        """
        Check if my object is changed
        :param keys: The keys that will determine if an object is changed
        :param module: The module
        :param result: The result from the query
        :return:
        """
        for key in keys:
            if module.params.get(key) != result[key]:
                return True
        return False