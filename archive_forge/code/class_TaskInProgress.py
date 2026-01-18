import logging
from oslo_vmware._i18n import _
class TaskInProgress(VimException):
    msg_fmt = _('Entity has another operation in process.')