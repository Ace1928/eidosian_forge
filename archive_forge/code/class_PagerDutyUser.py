from __future__ import absolute_import, division, print_function
from os import path
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
class PagerDutyUser(object):

    def __init__(self, module, session):
        self._module = module
        self._apisession = session

    def does_user_exist(self, pd_email):
        for user in self._apisession.iter_all('users'):
            if user['email'] == pd_email:
                return user['id']

    def add_pd_user(self, pd_name, pd_email, pd_role):
        try:
            user = self._apisession.persist('users', 'email', {'name': pd_name, 'email': pd_email, 'type': 'user', 'role': pd_role})
            return user
        except PDClientError as e:
            if e.response.status_code == 400:
                self._module.fail_json(msg='Failed to add %s due to invalid argument' % pd_name)
            if e.response.status_code == 401:
                self._module.fail_json(msg='Failed to add %s due to invalid API key' % pd_name)
            if e.response.status_code == 402:
                self._module.fail_json(msg='Failed to add %s due to inability to perform the action within the API token' % pd_name)
            if e.response.status_code == 403:
                self._module.fail_json(msg='Failed to add %s due to inability to review the requested resource within the API token' % pd_name)
            if e.response.status_code == 429:
                self._module.fail_json(msg='Failed to add %s due to reaching the limit of making requests' % pd_name)

    def delete_user(self, pd_user_id, pd_name):
        try:
            user_path = path.join('/users/', pd_user_id)
            self._apisession.rdelete(user_path)
        except PDClientError as e:
            if e.response.status_code == 404:
                self._module.fail_json(msg='Failed to remove %s as user was not found' % pd_name)
            if e.response.status_code == 403:
                self._module.fail_json(msg='Failed to remove %s due to inability to review the requested resource within the API token' % pd_name)
            if e.response.status_code == 401:
                pd_incidents = self.get_incidents_assigned_to_user(pd_user_id)
                self._module.fail_json(msg='Failed to remove %s as user has assigned incidents %s' % (pd_name, pd_incidents))
            if e.response.status_code == 429:
                self._module.fail_json(msg='Failed to remove %s due to reaching the limit of making requests' % pd_name)

    def get_incidents_assigned_to_user(self, pd_user_id):
        incident_info = {}
        incidents = self._apisession.list_all('incidents', params={'user_ids[]': [pd_user_id]})
        for incident in incidents:
            incident_info = {'title': incident['title'], 'key': incident['incident_key'], 'status': incident['status']}
        return incident_info

    def add_user_to_teams(self, pd_user_id, pd_teams, pd_role):
        updated_team = None
        for team in pd_teams:
            team_info = self._apisession.find('teams', team, attribute='name')
            if team_info is not None:
                try:
                    updated_team = self._apisession.rput('/teams/' + team_info['id'] + '/users/' + pd_user_id, json={'role': pd_role})
                except PDClientError:
                    updated_team = None
        return updated_team