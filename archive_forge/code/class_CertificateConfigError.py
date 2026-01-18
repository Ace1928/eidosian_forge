from keystoneauth1 import exceptions as _exc
from keystoneclient.i18n import _
from_response = _exc.from_response
class CertificateConfigError(Exception):
    """Error reading the certificate."""

    def __init__(self, output):
        self.output = output
        msg = _('Unable to load certificate.')
        super(CertificateConfigError, self).__init__(msg)