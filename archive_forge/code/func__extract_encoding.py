import email.message
import os
from configparser import RawConfigParser
from .cmd import Command
def _extract_encoding(content_type):
    """
    >>> _extract_encoding('text/plain')
    'ascii'
    >>> _extract_encoding('text/html; charset="utf8"')
    'utf8'
    """
    msg = email.message.EmailMessage()
    msg['content-type'] = content_type
    return msg['content-type'].params.get('charset', 'ascii')