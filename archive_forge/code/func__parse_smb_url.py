from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
def _parse_smb_url(url):
    e = 'SMB url must be smb://workgroup;user:password@server:port/share/folder/file.txt: '
    if not url:
        raise ValueError('SMB error: no host given')
    if not url.startswith('smb://'):
        raise ValueError(e + url)
    if PY3:
        from urllib.parse import urlparse
    else:
        from urlparse import urlparse
    parsed = urlparse(url)
    if not parsed.path:
        raise ValueError(e + url)
    unc_path = parsed.path.replace('/', '\\')
    server_path = '\\\\{}{}'.format(parsed.hostname, unc_path)
    if not parsed.username:
        domain = None
        username = None
    elif ';' in parsed.username:
        domain, username = parsed.username.split(';')
    else:
        domain, username = (None, parsed.username)
    port = 445 if not parsed.port else int(parsed.port)
    return (domain, parsed.hostname, port, username, parsed.password, server_path)