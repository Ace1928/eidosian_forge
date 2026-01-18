import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
def _load_session_kwargs(self):
    return {'username': None, 'project_id': None, 'project_name': None, 'auth_url': None, 'password': None, 'auth_type': 'password', 'insecure': False, 'user_domain_id': None, 'user_domain_name': None, 'project_domain_id': None, 'project_domain_name': None, 'auth_token': None, 'timeout': 600}