from keystoneauth1 import exceptions as _exc
from keystoneclient.i18n import _
from_response = _exc.from_response
class CMSError(Exception):
    """Error reading the certificate."""

    def __init__(self, output):
        self.output = output
        msg = _('Unable to sign or verify data.')
        super(CMSError, self).__init__(msg)