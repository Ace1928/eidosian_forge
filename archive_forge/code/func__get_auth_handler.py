import logging
import warnings
from keystoneauth1 import identity
from keystoneauth1 import session as k_session
from monascaclient.osc import migration
from monascaclient import version
def _get_auth_handler(kwargs):
    if 'token' in kwargs:
        auth = identity.Token(auth_url=kwargs.get('auth_url', None), token=kwargs.get('token', None), project_id=kwargs.get('project_id', None), project_name=kwargs.get('project_name', None), project_domain_id=kwargs.get('project_domain_id', None), project_domain_name=kwargs.get('project_domain_name', None))
    elif {'username', 'password'} <= set(kwargs):
        auth = identity.Password(auth_url=kwargs.get('auth_url', None), username=kwargs.get('username', None), password=kwargs.get('password', None), project_id=kwargs.get('project_id', None), project_name=kwargs.get('project_name', None), project_domain_id=kwargs.get('project_domain_id', None), project_domain_name=kwargs.get('project_domain_name', None), user_domain_id=kwargs.get('user_domain_id', None), user_domain_name=kwargs.get('user_domain_name', None))
    else:
        raise Exception('monascaclient can be configured with either "token" or "username:password" but neither of them was found in passed arguments.')
    return auth