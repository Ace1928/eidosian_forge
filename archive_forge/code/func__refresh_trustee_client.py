from keystoneauth1 import exceptions as ka_exceptions
from keystoneauth1 import loading as ka_loading
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_log import log as logging
def _refresh_trustee_client(self):
    kwargs = {'project_name': None, 'project_domain_name': None, 'project_id': None, 'trust_id': self.trust_id}
    trustee_auth = ka_loading.load_auth_from_conf_options(CONF, 'keystone_authtoken', **kwargs)
    return self._load_client(trustee_auth)