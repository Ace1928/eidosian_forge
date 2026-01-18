import logging
from oslo_vmware._i18n import _
class VimAttributeException(VMwareDriverException):
    """Thrown when a particular attribute cannot be found."""

    def __init__(self, message, cause=None):
        super(VimAttributeException, self).__init__(message)
        self.cause = cause