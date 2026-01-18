from __future__ import (absolute_import, division, print_function)
import json
import os
import os.path
import stat
import tempfile
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.basic import FILE_COMMON_ARGUMENTS
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.hashing import checksum
def _create_content_tempfile(self, content):
    """ Create a tempfile containing defined content """
    fd, content_tempfile = tempfile.mkstemp(dir=C.DEFAULT_LOCAL_TMP)
    f = os.fdopen(fd, 'wb')
    content = to_bytes(content)
    try:
        f.write(content)
    except Exception as err:
        os.remove(content_tempfile)
        raise Exception(err)
    finally:
        f.close()
    return content_tempfile