from oslo_utils import encodeutils
from neutronclient._i18n import _
class RequestURITooLong(NeutronClientException):
    """Raised when a request fails with HTTP error 414."""

    def __init__(self, **kwargs):
        self.excess = kwargs.get('excess', 0)
        super(RequestURITooLong, self).__init__(**kwargs)