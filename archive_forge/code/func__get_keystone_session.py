import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ka_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import session
from zaqarclient.auth import base
from zaqarclient import errors
def _get_keystone_session(self, **kwargs):
    cacert = kwargs.pop('cacert', None)
    cert = kwargs.pop('cert', None)
    key = kwargs.pop('key', None)
    insecure = kwargs.pop('insecure', False)
    auth_url = kwargs.pop('auth_url', None)
    project_id = kwargs.pop('project_id', None)
    project_name = kwargs.pop('project_name', None)
    token = kwargs.get('token')
    if insecure:
        verify = False
    else:
        verify = cacert or True
    if cert and key:
        cert = (cert, key)
    ks_session = session.Session(verify=verify, cert=cert)
    v2_auth_url, v3_auth_url = self._discover_auth_versions(ks_session, auth_url)
    username = kwargs.pop('username', None)
    user_id = kwargs.pop('user_id', None)
    user_domain_name = kwargs.pop('user_domain_name', None)
    user_domain_id = kwargs.pop('user_domain_id', None)
    project_domain_name = kwargs.pop('project_domain_name', None)
    project_domain_id = kwargs.pop('project_domain_id', None)
    auth = None
    use_domain = user_domain_id or user_domain_name or project_domain_id or project_domain_name
    use_v3 = v3_auth_url and (use_domain or not v2_auth_url)
    use_v2 = v2_auth_url and (not use_domain)
    if use_v3 and token:
        auth = v3_auth.Token(v3_auth_url, token=token, project_name=project_name, project_id=project_id, project_domain_name=project_domain_name, project_domain_id=project_domain_id)
    elif use_v2 and token:
        auth = v2_auth.Token(v2_auth_url, token=token, tenant_id=project_id, tenant_name=project_name)
    elif use_v3:
        auth = v3_auth.Password(v3_auth_url, username=username, password=kwargs.pop('password', None), user_id=user_id, user_domain_name=user_domain_name, user_domain_id=user_domain_id, project_name=project_name, project_id=project_id, project_domain_name=project_domain_name, project_domain_id=project_domain_id)
    elif use_v2:
        auth = v2_auth.Password(v2_auth_url, username, kwargs.pop('password', None), tenant_id=project_id, tenant_name=project_name)
    else:
        raise errors.ZaqarError('Unable to determine the Keystone version to authenticate with using the given auth_url.')
    ks_session.auth = auth
    return ks_session