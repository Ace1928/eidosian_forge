import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
class _BaseSAMLPlugin(v3.AuthConstructor):
    HTTP_MOVED_TEMPORARILY = 302
    HTTP_SEE_OTHER = 303
    PROTOCOL = 'saml2'

    @staticmethod
    def _first(_list):
        if len(_list) != 1:
            raise IndexError(_('Only single element list is acceptable'))
        return _list[0]

    @staticmethod
    def str_to_xml(content, msg=None, include_exc=True):
        try:
            return etree.XML(content)
        except etree.XMLSyntaxError as e:
            if not msg:
                msg = str(e)
            else:
                msg = msg % e if include_exc else msg
            raise exceptions.AuthorizationFailure(msg)

    @staticmethod
    def xml_to_str(content, **kwargs):
        return etree.tostring(content, **kwargs)

    @property
    def token_url(self):
        """Return full URL where authorization data is sent."""
        values = {'host': self.auth_url.rstrip('/'), 'identity_provider': self.identity_provider, 'protocol': self.PROTOCOL}
        url = '%(host)s/OS-FEDERATION/identity_providers/%(identity_provider)s/protocols/%(protocol)s/auth'
        url = url % values
        return url

    @classmethod
    def get_options(cls):
        options = super(_BaseSAMLPlugin, cls).get_options()
        options.extend([cfg.StrOpt('identity-provider', help="Identity Provider's name"), cfg.StrOpt('identity-provider-url', help="Identity Provider's URL"), cfg.StrOpt('username', dest='username', help='Username', deprecated_name='user-name'), cfg.StrOpt('password', secret=True, help='Password')])
        return options