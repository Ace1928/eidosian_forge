import logging
from oslo_vmware._i18n import _
class ManagedObjectNotFoundException(VimException):
    msg_fmt = _('Managed object not found.')
    code = 404