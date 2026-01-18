import logging
from oslo_vmware._i18n import _
class ToolsUnavailableException(VimException):
    msg_fmt = _('VMware Tools is not running.')