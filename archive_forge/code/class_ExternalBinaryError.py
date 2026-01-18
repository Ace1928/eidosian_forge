from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
class ExternalBinaryError(Exception):
    """Exception raised when gsutil runs an external binary, and it fails."""

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __repr__(self):
        return 'ExternalBinaryError: %s' % self.message