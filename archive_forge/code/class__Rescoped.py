import abc
from keystoneauth1.identity.v3 import base
from keystoneauth1.identity.v3 import token
class _Rescoped(base.BaseAuth, metaclass=abc.ABCMeta):
    """A plugin that is always going to go through a rescope process.

    The original keystone plugins could simply pass a project or domain to
    along with the credentials and get a scoped token. For federation, K2K and
    newer mechanisms we always get an unscoped token first and then rescope.

    This is currently not public as it's generally an abstraction of a flow
    used by plugins within keystoneauth1.

    It also cannot go in base as it depends on token.Token for rescoping which
    would create a circular dependency.
    """
    rescoping_plugin = token.Token

    def _get_scoping_data(self):
        return {'trust_id': self.trust_id, 'domain_id': self.domain_id, 'domain_name': self.domain_name, 'project_id': self.project_id, 'project_name': self.project_name, 'project_domain_id': self.project_domain_id, 'project_domain_name': self.project_domain_name}

    def get_auth_ref(self, session, **kwargs):
        """Authenticate retrieve token information.

        This is a multi-step process where a client does federated authn
        receives an unscoped token.

        If an unscoped token is successfully received and scoping information
        is present then the token is rescoped to that target.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a token data representation
        :rtype: :py:class:`keystoneauth1.access.AccessInfo`

        """
        auth_ref = self.get_unscoped_auth_ref(session)
        scoping = self._get_scoping_data()
        if any(scoping.values()):
            token_plugin = self.rescoping_plugin(self.auth_url, token=auth_ref.auth_token, **scoping)
            auth_ref = token_plugin.get_auth_ref(session)
        return auth_ref

    @abc.abstractmethod
    def get_unscoped_auth_ref(self, session, **kwargs):
        """Fetch unscoped federated token."""