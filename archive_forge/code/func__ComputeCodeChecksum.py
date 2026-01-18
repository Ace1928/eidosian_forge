from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import sys
import six
import boto
import crcmod
import gslib
from gslib.command import Command
from gslib.utils import system_util
from gslib.utils.boto_util import GetFriendlyConfigFilePaths
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import UTF8
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
def _ComputeCodeChecksum(self):
    """Computes a checksum of gsutil code.

    This checksum can be used to determine if users locally modified
    gsutil when requesting support. (It's fine for users to make local mods,
    but when users ask for support we ask them to run a stock version of
    gsutil so we can reduce possible variables.)

    Returns:
      MD5 checksum of gsutil code.
    """
    if gslib.IS_PACKAGE_INSTALL:
        return 'PACKAGED_GSUTIL_INSTALLS_DO_NOT_HAVE_CHECKSUMS'
    m = GetMd5()
    files_to_checksum = [gslib.GSUTIL_PATH]
    for root, _, files in os.walk(gslib.GSLIB_DIR):
        for filepath in files:
            if filepath.endswith('.py'):
                files_to_checksum.append(os.path.join(root, filepath))
    for filepath in sorted(files_to_checksum):
        if six.PY2:
            f = open(filepath, 'rb')
            content = f.read()
            content = re.sub('(\\r\\n|\\r|\\n)', b'\n', content)
            m.update(content)
            f.close()
        else:
            f = open(filepath, 'r', encoding=UTF8)
            content = f.read()
            content = re.sub('(\\r\\n|\\r|\\n)', '\n', content)
            m.update(content.encode(UTF8))
            f.close()
    return m.hexdigest()