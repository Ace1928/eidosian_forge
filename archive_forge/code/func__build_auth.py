from keystone.i18n import _
def _build_auth(self, user_id=None, username=None, user_domain_id=None, user_domain_name=None, **kwargs):
    self.assertEqual(1, len(kwargs), message='_build_auth requires 1 (and only 1) secret type and value')
    secret_type, secret_value = list(kwargs.items())[0]
    self.assertIn(secret_type, ('passcode', 'password'), message="_build_auth only supports 'passcode' and 'password' secret types")
    data = {}
    data['user'] = self._build_user(user_id=user_id, username=username, user_domain_id=user_domain_id, user_domain_name=user_domain_name)
    data['user'][secret_type] = secret_value
    return data