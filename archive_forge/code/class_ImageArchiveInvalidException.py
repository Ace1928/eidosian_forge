from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
from ansible.module_utils.common.text.converters import to_native
class ImageArchiveInvalidException(Exception):

    def __init__(self, message, cause):
        """
        :param message: Exception message
        :type message: str
        :param cause: Inner exception that this exception wraps
        :type cause: Exception | None
        """
        super(ImageArchiveInvalidException, self).__init__(message)
        self.cause = cause