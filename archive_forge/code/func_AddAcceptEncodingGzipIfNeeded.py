from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def AddAcceptEncodingGzipIfNeeded(headers_dict, compressed_encoding=False):
    if compressed_encoding:
        headers_dict['accept-encoding'] = 'gzip'