from keystoneauth1 import exceptions as ka_exceptions
from keystoneauth1 import loading as ka_loading
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_log import log as logging
class TokenRefresher(object):
    """Class that responsible for token refreshing with trusts"""

    def __init__(self, user_plugin, user_project, user_roles):
        """Prepare all parameters and clients required to refresh token"""
        trustor_client = self._load_client(user_plugin)
        trustor_id = trustor_client.session.get_user_id()
        trustee_user_auth = ka_loading.load_auth_from_conf_options(CONF, 'keystone_authtoken')
        self.trustee_user_client = self._load_client(trustee_user_auth)
        trustee_id = self.trustee_user_client.session.get_user_id()
        self.trust_id = trustor_client.trusts.create(trustor_user=trustor_id, trustee_user=trustee_id, impersonation=True, role_names=user_roles, project=user_project).id
        LOG.debug('Trust %s has been created.', self.trust_id)
        self.trustee_client = None

    def refresh_token(self):
        """Receive new token if user need to update old token

        :return: new token that can be used for authentication
        """
        LOG.debug('Requesting the new token with trust %s', self.trust_id)
        if self.trustee_client is None:
            self.trustee_client = self._refresh_trustee_client()
        try:
            return self.trustee_client.session.get_token()
        except ka_exceptions.Unauthorized:
            self.trustee_client = self._refresh_trustee_client()
            return self.trustee_client.session.get_token()

    def release_resources(self):
        """Release keystone resources required for refreshing"""
        try:
            if self.trustee_client is None:
                self._refresh_trustee_client().trusts.delete(self.trust_id)
            else:
                self.trustee_client.trusts.delete(self.trust_id)
        except ka_exceptions.Unauthorized:
            self.trustee_client = self._refresh_trustee_client()
            self.trustee_client.trusts.delete(self.trust_id)

    def _refresh_trustee_client(self):
        kwargs = {'project_name': None, 'project_domain_name': None, 'project_id': None, 'trust_id': self.trust_id}
        trustee_auth = ka_loading.load_auth_from_conf_options(CONF, 'keystone_authtoken', **kwargs)
        return self._load_client(trustee_auth)

    @staticmethod
    def _load_client(plugin):
        sess = ka_loading.load_session_from_conf_options(CONF, 'keystone_authtoken', auth=plugin)
        return ks_client.Client(session=sess)