import logging
from oslo_vmware._i18n import _
class NotAuthenticatedException(VimException):
    msg_fmt = _('Not Authenticated.')
    code = 403