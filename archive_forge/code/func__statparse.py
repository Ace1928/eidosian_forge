import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _statparse(self, resp):
    """Internal: parse the response line of a STAT, NEXT, LAST,
        ARTICLE, HEAD or BODY command."""
    if not resp.startswith('22'):
        raise NNTPReplyError(resp)
    words = resp.split()
    art_num = int(words[1])
    message_id = words[2]
    return (resp, art_num, message_id)