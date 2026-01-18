from keystone.i18n import _
def _build_app_cred_auth(self, secret, app_cred_id=None, app_cred_name=None, user_id=None, username=None, user_domain_id=None, user_domain_name=None):
    data = {'secret': secret}
    if app_cred_id:
        data['id'] = app_cred_id
    else:
        data['name'] = app_cred_name
        data['user'] = self._build_user(user_id=user_id, username=username, user_domain_id=user_domain_id, user_domain_name=user_domain_name)
    return data