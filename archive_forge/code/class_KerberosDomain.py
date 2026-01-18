import abc
import flask
from keystone.auth.plugins import base
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
class KerberosDomain(Domain):
    """Allows `kerberos` as a method."""

    def _authenticate(self):
        if flask.request.environ.get('AUTH_TYPE') != 'Negotiate':
            raise exception.Unauthorized(_('auth_type is not Negotiate'))
        return super(KerberosDomain, self)._authenticate()