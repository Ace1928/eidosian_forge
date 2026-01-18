import hashlib
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
def _build_idp_id(self):
    """Build the IdP name from the given config option issuer_attribute.

        The default issuer attribute SSL_CLIENT_I_DN in the environment is
        built with the following formula -

        base64_idp = sha1(env['SSL_CLIENT_I_DN'])

        :returns: base64_idp like the above example
        :rtype: str
        """
    idp = self.env.get(CONF.tokenless_auth.issuer_attribute)
    if idp is None:
        raise exception.TokenlessAuthConfigError(issuer_attribute=CONF.tokenless_auth.issuer_attribute)
    hashed_idp = hashlib.sha256(idp.encode('utf-8'))
    return hashed_idp.hexdigest()