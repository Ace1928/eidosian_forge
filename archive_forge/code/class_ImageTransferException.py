import logging
from oslo_vmware._i18n import _
class ImageTransferException(VMwareDriverException):
    """Thrown when there is an error during image transfer."""

    def __init__(self, message, cause=None):
        super(ImageTransferException, self).__init__(message)
        self.cause = cause