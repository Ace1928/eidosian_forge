import abc
import flask
from keystone.auth.plugins import base
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
class DefaultDomain(Base):

    def _authenticate(self):
        """Use remote_user to look up the user in the identity backend."""
        return PROVIDERS.identity_api.get_user_by_name(flask.request.remote_user, CONF.identity.default_domain_id)