import os
from keystoneauth1 import loading
from keystoneauth1 import plugin
class CinderNoAuthPlugin(plugin.BaseAuthPlugin):

    def __init__(self, user_id, project_id=None, roles=None, endpoint=None):
        self._user_id = user_id
        self._project_id = project_id if project_id else user_id
        self._endpoint = endpoint
        self._roles = roles
        self.auth_token = '%s:%s' % (self._user_id, self._project_id)

    def get_headers(self, session, **kwargs):
        return {'x-user-id': self._user_id, 'x-project-id': self._project_id, 'X-Auth-Token': self.auth_token}

    def get_user_id(self, session, **kwargs):
        return self._user_id

    def get_project_id(self, session, **kwargs):
        return self._project_id

    def get_endpoint(self, session, **kwargs):
        return '%s/%s' % (self._endpoint, self._project_id)

    def invalidate(self):
        pass