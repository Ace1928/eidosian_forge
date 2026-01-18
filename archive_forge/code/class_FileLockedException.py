import logging
from oslo_vmware._i18n import _
class FileLockedException(VimException):
    msg_fmt = _('File locked.')
    code = 403