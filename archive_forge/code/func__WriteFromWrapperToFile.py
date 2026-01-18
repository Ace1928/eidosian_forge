from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import os
import pkgutil
import six
import gslib.cloud_api
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
def _WriteFromWrapperToFile(self, daisy_chain_wrapper, file_path):
    """Writes all contents from the DaisyChainWrapper to the named file."""
    with open(file_path, 'wb') as upload_stream:
        while True:
            data = daisy_chain_wrapper.read(TRANSFER_BUFFER_SIZE)
            if not data:
                break
            upload_stream.write(data)