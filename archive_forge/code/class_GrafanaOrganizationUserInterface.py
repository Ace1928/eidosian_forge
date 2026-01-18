from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
class GrafanaOrganizationUserInterface(object):

    def __init__(self, module):
        self._module = module
        self.headers = {'Content-Type': 'application/json'}
        self.headers['Authorization'] = basic_auth_header(module.params['url_username'], module.params['url_password'])
        self.grafana_url = clean_url(module.params.get('url'))

    def _api_call(self, method, path, payload):
        data = None
        if payload:
            data = json.dumps(payload)
        return fetch_url(self._module, self.grafana_url + '/api/' + path, headers=self.headers, method=method, data=data)

    def _organization_by_name(self, org_name):
        r, info = self._api_call('GET', 'orgs/name/%s' % org_name, None)
        if info['status'] != 200:
            raise GrafanaAPIException('Unable to retrieve organization: %s' % info)
        return json.loads(to_text(r.read()))

    def _organization_users(self, org_id):
        r, info = self._api_call('GET', 'orgs/%d/users' % org_id, None)
        if info['status'] != 200:
            raise GrafanaAPIException('Unable to retrieve organization users: %s' % info)
        return json.loads(to_text(r.read()))

    def _create_organization_user(self, org_id, login, role):
        return self._api_call('POST', 'orgs/%d/users' % org_id, {'loginOrEmail': login, 'role': role})

    def _update_organization_user_role(self, org_id, user_id, role):
        return self._api_call('PATCH', 'orgs/%d/users/%s' % (org_id, user_id), {'role': role})

    def _remove_organization_user(self, org_id, user_id):
        return self._api_call('DELETE', 'orgs/%d/users/%s' % (org_id, user_id), None)

    def _organization_user_by_login(self, org_id, login):
        for user in self._organization_users(org_id):
            if login in (user['login'], user['email']):
                return user

    def create_or_update_user(self, org_id, login, role):
        r, info = self._create_organization_user(org_id, login, role)
        if info['status'] == 200:
            return {'state': 'present', 'changed': True, 'user': self._organization_user_by_login(org_id, login)}
        if info['status'] == 409:
            user = self._organization_user_by_login(org_id, login)
            if not user:
                raise Exception('[BUG] User not found in organization')
            if user['role'] == role:
                return {'changed': False}
            r, info = self._update_organization_user_role(org_id, user['userId'], role)
            if info['status'] == 200:
                return {'changed': True, 'user': self._organization_user_by_login(org_id, login)}
            else:
                raise GrafanaAPIException('Unable to update organization user: %s' % info)
        else:
            raise GrafanaAPIException('Unable to add user to organization: %s' % info)

    def remove_user(self, org_id, login):
        user = self._organization_user_by_login(org_id, login)
        if not user:
            return {'changed': False}
        r, info = self._remove_organization_user(org_id, user['userId'])
        if info['status'] == 200:
            return {'state': 'absent', 'changed': True}
        else:
            raise GrafanaAPIException('Unable to delete organization user: %s' % info)