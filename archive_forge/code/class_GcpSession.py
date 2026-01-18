from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
class GcpSession(object):

    def __init__(self, module, product):
        self.module = module
        self.product = product
        self._validate()

    def get(self, url, body=None, **kwargs):
        """
        This method should be avoided in favor of full_get
        """
        kwargs.update({'json': body})
        return self.full_get(url, **kwargs)

    def post(self, url, body=None, headers=None, **kwargs):
        """
        This method should be avoided in favor of full_post
        """
        kwargs.update({'json': body, 'headers': headers})
        return self.full_post(url, **kwargs)

    def post_contents(self, url, file_contents=None, headers=None, **kwargs):
        """
        This method should be avoided in favor of full_post
        """
        kwargs.update({'data': file_contents, 'headers': headers})
        return self.full_post(url, **kwargs)

    def delete(self, url, body=None):
        """
        This method should be avoided in favor of full_delete
        """
        kwargs = {'json': body}
        return self.full_delete(url, **kwargs)

    def put(self, url, body=None, params=None):
        """
        This method should be avoided in favor of full_put
        """
        kwargs = {'json': body}
        return self.full_put(url, params=params, **kwargs)

    def patch(self, url, body=None, **kwargs):
        """
        This method should be avoided in favor of full_patch
        """
        kwargs.update({'json': body})
        return self.full_patch(url, **kwargs)

    def list(self, url, callback, params=None, array_name='items', pageToken='nextPageToken', **kwargs):
        """
        This should be used for calling the GCP list APIs. It will return
        an array of items

        This takes a callback to a `return_if_object(module, response)`
        function that will decode the response + return a dictionary. Some
        modules handle the decode + error processing differently, so we should
        defer to the module to handle this.
        """
        resp = callback(self.module, self.full_get(url, params, **kwargs))
        items = resp.get(array_name) if resp.get(array_name) else []
        while resp.get(pageToken):
            if params:
                params['pageToken'] = resp.get(pageToken)
            else:
                params = {'pageToken': resp[pageToken]}
            resp = callback(self.module, self.full_get(url, params, **kwargs))
            if resp.get(array_name):
                items = items + resp.get(array_name)
        return items

    def full_get(self, url, params=None, **kwargs):
        kwargs['headers'] = self._set_headers(kwargs.get('headers'))
        try:
            return self.session().get(url, params=params, **kwargs)
        except getattr(requests.exceptions, 'RequestException') as inst:
            self.module.fail_json(msg=inst.message)

    def full_post(self, url, data=None, json=None, **kwargs):
        kwargs['headers'] = self._set_headers(kwargs.get('headers'))
        try:
            return self.session().post(url, data=data, json=json, **kwargs)
        except getattr(requests.exceptions, 'RequestException') as inst:
            self.module.fail_json(msg=inst.message)

    def full_put(self, url, data=None, **kwargs):
        kwargs['headers'] = self._set_headers(kwargs.get('headers'))
        try:
            return self.session().put(url, data=data, **kwargs)
        except getattr(requests.exceptions, 'RequestException') as inst:
            self.module.fail_json(msg=inst.message)

    def full_patch(self, url, data=None, **kwargs):
        kwargs['headers'] = self._set_headers(kwargs.get('headers'))
        try:
            return self.session().patch(url, data=data, **kwargs)
        except getattr(requests.exceptions, 'RequestException') as inst:
            self.module.fail_json(msg=inst.message)

    def full_delete(self, url, **kwargs):
        kwargs['headers'] = self._set_headers(kwargs.get('headers'))
        try:
            return self.session().delete(url, **kwargs)
        except getattr(requests.exceptions, 'RequestException') as inst:
            self.module.fail_json(msg=inst.message)

    def _set_headers(self, headers):
        if headers:
            return self._merge_dictionaries(headers, self._headers())
        return self._headers()

    def session(self):
        return AuthorizedSession(self._credentials())

    def _validate(self):
        if not HAS_REQUESTS:
            self.module.fail_json(msg='Please install the requests library')
        if not HAS_GOOGLE_LIBRARIES:
            self.module.fail_json(msg='Please install the google-auth library')
        if self.module.params.get('service_account_email') is not None and self.module.params['auth_kind'] != 'machineaccount':
            self.module.fail_json(msg='Service Account Email only works with Machine Account-based authentication')
        if (self.module.params.get('service_account_file') is not None or self.module.params.get('service_account_contents') is not None) and self.module.params['auth_kind'] != 'serviceaccount':
            self.module.fail_json(msg='Service Account File only works with Service Account-based authentication')
        if self.module.params.get('access_token') is not None and self.module.params['auth_kind'] != 'accesstoken':
            self.module.fail_json(msg='Supplying access_token requires auth_kind set to accesstoken')

    def _credentials(self):
        cred_type = self.module.params['auth_kind']
        if cred_type == 'application':
            credentials, project_id = google.auth.default(scopes=self.module.params['scopes'])
            return credentials
        if cred_type == 'serviceaccount':
            service_account_file = self.module.params.get('service_account_file')
            service_account_contents = self.module.params.get('service_account_contents')
            if service_account_file is not None:
                path = os.path.realpath(os.path.expanduser(service_account_file))
                try:
                    svc_acct_creds = service_account.Credentials.from_service_account_file(path)
                except OSError as e:
                    self.module.fail_json(msg='Unable to read service_account_file at %s: %s' % (path, e.strerror))
            elif service_account_contents is not None:
                try:
                    info = json.loads(service_account_contents)
                except json.decoder.JSONDecodeError as e:
                    self.module.fail_json(msg='Unable to decode service_account_contents as JSON: %s' % e)
                svc_acct_creds = service_account.Credentials.from_service_account_info(info)
            else:
                self.module.fail_json(msg='Service Account authentication requires setting either service_account_file or service_account_contents')
            return svc_acct_creds.with_scopes(self.module.params['scopes'])
        if cred_type == 'machineaccount':
            email = self.module.params['service_account_email']
            email = email if email is not None else 'default'
            return google.auth.compute_engine.Credentials(email)
        if cred_type == 'accesstoken':
            access_token = self.module.params['access_token']
            if access_token is None:
                self.module.fail_json(msg='An access token must be supplied when auth_kind is accesstoken')
            return oauth2.Credentials(access_token, scopes=self.module.params['scopes'])
        self.module.fail_json(msg="Credential type '%s' not implemented" % cred_type)

    def _headers(self):
        user_agent = 'Google-Ansible-MM-{0}'.format(self.product)
        if self.module.params.get('env_type'):
            user_agent = '{0}-{1}'.format(user_agent, self.module.params.get('env_type'))
        return {'User-Agent': user_agent}

    def _merge_dictionaries(self, a, b):
        new = a.copy()
        new.update(b)
        return new