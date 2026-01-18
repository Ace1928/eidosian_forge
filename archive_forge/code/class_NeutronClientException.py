from oslo_utils import encodeutils
from neutronclient._i18n import _
class NeutronClientException(NeutronException):
    """Base exception which exceptions from Neutron are mapped into.

    NOTE: on the client side, we use different exception types in order
    to allow client library users to handle server exceptions in try...except
    blocks. The actual error message is the one generated on the server side.
    """
    status_code = 0
    req_ids_msg = _('Neutron server returns request_ids: %s')
    request_ids = []

    def __init__(self, message=None, **kwargs):
        self.request_ids = kwargs.get('request_ids')
        if 'status_code' in kwargs:
            self.status_code = kwargs['status_code']
        if self.request_ids:
            req_ids_msg = self.req_ids_msg % self.request_ids
            if message:
                message = _('%(msg)s\n%(id)s') % {'msg': message, 'id': req_ids_msg}
            else:
                message = req_ids_msg
        super(NeutronClientException, self).__init__(message, **kwargs)