from __future__ import (absolute_import, division, print_function)
import codecs
import os
import os.path
import re
import tempfile
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAction, _AnsibleActionDone, AnsibleActionFail
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.hashing import checksum_s
def _assemble_from_fragments(self, src_path, delimiter=None, compiled_regexp=None, ignore_hidden=False, decrypt=True):
    """ assemble a file from a directory of fragments """
    tmpfd, temp_path = tempfile.mkstemp(dir=C.DEFAULT_LOCAL_TMP)
    tmp = os.fdopen(tmpfd, 'wb')
    delimit_me = False
    add_newline = False
    for f in (to_text(p, errors='surrogate_or_strict') for p in sorted(os.listdir(src_path))):
        if compiled_regexp and (not compiled_regexp.search(f)):
            continue
        fragment = u'%s/%s' % (src_path, f)
        if not os.path.isfile(fragment) or (ignore_hidden and os.path.basename(fragment).startswith('.')):
            continue
        with open(self._loader.get_real_file(fragment, decrypt=decrypt), 'rb') as fragment_fh:
            fragment_content = fragment_fh.read()
        if add_newline:
            tmp.write(b'\n')
        if delimit_me:
            if delimiter:
                delimiter = codecs.escape_decode(delimiter)[0]
                tmp.write(delimiter)
                if delimiter[-1] != b'\n':
                    tmp.write(b'\n')
        tmp.write(fragment_content)
        delimit_me = True
        if fragment_content.endswith(b'\n'):
            add_newline = False
        else:
            add_newline = True
    tmp.close()
    return temp_path