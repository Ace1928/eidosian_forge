import logging
from oslo_vmware._i18n import _
class DuplicateName(VimException):
    msg_fmt = _('Duplicate name.')