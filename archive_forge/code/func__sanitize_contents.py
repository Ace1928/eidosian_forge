from __future__ import absolute_import, division, print_function
import os
import re
import time
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action.normal import ActionModule as _ActionModule
from ansible.utils.display import Display
from ansible.utils.hashing import checksum, checksum_s
def _sanitize_contents(self, contents, filters):
    """remove lines from contents that match
        regexes specified in the `filters` list
        """
    for x in filters:
        contents = re.sub(x, '', contents)
    return contents.strip()