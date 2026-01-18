import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _getdescriptions(self, group_pattern, return_all):
    line_pat = re.compile('^(?P<group>[^ \t]+)[ \t]+(.*)$')
    resp, lines = self._longcmdstring('LIST NEWSGROUPS ' + group_pattern)
    if not resp.startswith('215'):
        resp, lines = self._longcmdstring('XGTITLE ' + group_pattern)
    groups = {}
    for raw_line in lines:
        match = line_pat.search(raw_line.strip())
        if match:
            name, desc = match.group(1, 2)
            if not return_all:
                return desc
            groups[name] = desc
    if return_all:
        return (resp, groups)
    else:
        return ''