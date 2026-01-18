import logging
from oslo_vmware._i18n import _
class FileAlreadyExistsException(VimException):
    msg_fmt = _('File already exists.')
    code = 409