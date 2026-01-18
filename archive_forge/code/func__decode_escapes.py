from __future__ import (absolute_import, division, print_function)
import codecs
import re
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.quoting import unquote
def _decode_escapes(s):

    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')
    return _ESCAPE_SEQUENCE_RE.sub(decode_match, s)