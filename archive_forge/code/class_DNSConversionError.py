from __future__ import (absolute_import, division, print_function)
class DNSConversionError(Exception):

    def __init__(self, message):
        super(DNSConversionError, self).__init__(message)
        self.error_message = message