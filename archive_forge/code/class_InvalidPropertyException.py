import logging
from oslo_vmware._i18n import _
class InvalidPropertyException(VimException):
    msg_fmt = _('Invalid property.')
    code = 400