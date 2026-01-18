import logging
from oslo_vmware._i18n import _
class FileFaultException(VimException):
    msg_fmt = _('File fault.')
    code = 409