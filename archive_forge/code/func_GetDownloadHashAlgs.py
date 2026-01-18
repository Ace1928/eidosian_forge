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
def GetDownloadHashAlgs(logger, consider_md5=False, consider_crc32c=False):
    """Returns a dict of hash algorithms for validating an object.

  Args:
    logger: logging.Logger for outputting log messages.
    consider_md5: If True, consider using a md5 hash.
    consider_crc32c: If True, consider using a crc32c hash.

  Returns:
    Dict of (string, hash algorithm).

  Raises:
    CommandException if hash algorithms satisfying the boto config file
    cannot be returned.
  """
    check_hashes_config = config.get('GSUtil', 'check_hashes', CHECK_HASH_IF_FAST_ELSE_FAIL)
    if check_hashes_config == CHECK_HASH_NEVER:
        return {}
    hash_algs = {}
    if consider_md5:
        hash_algs['md5'] = GetMd5
    elif consider_crc32c:
        if UsingCrcmodExtension():
            hash_algs['crc32c'] = lambda: crcmod.predefined.Crc('crc-32c')
        elif not hash_algs:
            if check_hashes_config == CHECK_HASH_IF_FAST_ELSE_FAIL:
                raise CommandException(_SLOW_CRC_EXCEPTION_TEXT)
            elif check_hashes_config == CHECK_HASH_IF_FAST_ELSE_SKIP:
                logger.warn(_NO_HASH_CHECK_WARNING)
            elif check_hashes_config == CHECK_HASH_ALWAYS:
                logger.warn(_SLOW_CRCMOD_DOWNLOAD_WARNING)
                hash_algs['crc32c'] = lambda: crcmod.predefined.Crc('crc-32c')
            else:
                raise CommandException("Your boto config 'check_hashes' option is misconfigured.")
    return hash_algs