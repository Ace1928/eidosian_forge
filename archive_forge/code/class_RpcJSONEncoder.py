import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
class RpcJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, bytes):
            return {'__bytes__': base64.b64encode(o).decode('ascii')}
        if isinstance(o, wrapper.NoFilterMatched):
            return {'__exception__': 'NoFilterMatched'}
        elif isinstance(o, wrapper.FilterMatchNotExecutable):
            return {'__exception__': 'FilterMatchNotExecutable', 'match': o.match}
        else:
            return super(RpcJSONEncoder, self).default(o)