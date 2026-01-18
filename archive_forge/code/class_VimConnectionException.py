import logging
from oslo_vmware._i18n import _
class VimConnectionException(VMwareDriverException):
    """Thrown when there is a connection problem."""

    def __init__(self, message, cause=None):
        super(VimConnectionException, self).__init__(message)
        self.cause = cause