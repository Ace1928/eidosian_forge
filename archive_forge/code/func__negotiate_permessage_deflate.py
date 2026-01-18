import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
def _negotiate_permessage_deflate(self, extensions):
    if not extensions:
        return None
    deflate = extensions.get('permessage-deflate')
    if deflate is None:
        return None
    for config in deflate:
        want_config = {'server_no_context_takeover': config.get('server_no_context_takeover', False), 'client_no_context_takeover': config.get('client_no_context_takeover', False)}
        max_wbits = min(zlib.MAX_WBITS, 15)
        mwb = config.get('server_max_window_bits')
        if mwb is not None:
            if mwb is True:
                want_config['server_max_window_bits'] = max_wbits
            else:
                want_config['server_max_window_bits'] = int(config.get('server_max_window_bits', max_wbits))
                if not 8 <= want_config['server_max_window_bits'] <= 15:
                    continue
        mwb = config.get('client_max_window_bits')
        if mwb is not None:
            if mwb is True:
                want_config['client_max_window_bits'] = max_wbits
            else:
                want_config['client_max_window_bits'] = int(config.get('client_max_window_bits', max_wbits))
                if not 8 <= want_config['client_max_window_bits'] <= 15:
                    continue
        return want_config
    return None