from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import hashlib
import os
import six
from boto import config
import crcmod
from gslib.exception import CommandException
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
def _ReverseBits32(crc):
    return int('{0:032b}'.format(crc, width=32)[::-1], 2)