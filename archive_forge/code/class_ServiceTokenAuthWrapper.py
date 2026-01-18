from keystoneauth1 import plugin
class ServiceTokenAuthWrapper(plugin.BaseAuthPlugin):

    def __init__(self, user_auth, service_auth):
        super(ServiceTokenAuthWrapper, self).__init__()
        self.user_auth = user_auth
        self.service_auth = service_auth

    def get_headers(self, session, **kwargs):
        headers = self.user_auth.get_headers(session, **kwargs)
        token = self.service_auth.get_token(session, **kwargs)
        headers[SERVICE_AUTH_HEADER_NAME] = token
        return headers

    def invalidate(self):
        user = self.user_auth.invalidate()
        service = self.service_auth.invalidate()
        return user or service

    def get_connection_params(self, *args, **kwargs):
        params = self.service_auth.get_connection_params(*args, **kwargs)
        params.update(self.user_auth.get_connection_params(*args, **kwargs))
        return params

    def get_token(self, *args, **kwargs):
        return self.user_auth.get_token(*args, **kwargs)

    def get_endpoint(self, *args, **kwargs):
        return self.user_auth.get_endpoint(*args, **kwargs)

    def get_user_id(self, *args, **kwargs):
        return self.user_auth.get_user_id(*args, **kwargs)

    def get_project_id(self, *args, **kwargs):
        return self.user_auth.get_project_id(*args, **kwargs)

    def get_sp_auth_url(self, *args, **kwargs):
        return self.user_auth.get_sp_auth_url(*args, **kwargs)

    def get_sp_url(self, *args, **kwargs):
        return self.user_auth.get_sp_url(*args, **kwargs)