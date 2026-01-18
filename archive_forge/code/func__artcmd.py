import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _artcmd(self, line, file=None):
    """Internal: process a HEAD, BODY or ARTICLE command."""
    resp, lines = self._longcmd(line, file)
    resp, art_num, message_id = self._statparse(resp)
    return (resp, ArticleInfo(art_num, message_id, lines))