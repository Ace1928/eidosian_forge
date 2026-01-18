import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
def _internal_parser(self, sequence, fields=None, strict=None):

    def wanted_field(f):
        return fields is None or f in fields
    if isinstance(sequence, (str, bytes)):
        sequence = sequence.splitlines()
    curkey = None
    content = ''
    for linebytes in self._gpg_stripped_paragraph(self._skip_useless_lines(sequence), strict):
        line = self.decoder.decode_bytes(linebytes)
        m = self._new_field_re.match(line)
        if m:
            if curkey:
                self[curkey] = content
            curkey = m.group('key')
            if not wanted_field(curkey):
                curkey = None
                continue
            content = m.group('data')
            continue
        if line and line[0].isspace() and (not line.isspace()):
            content += '\n' + line
            continue
    if curkey:
        self[curkey] = content