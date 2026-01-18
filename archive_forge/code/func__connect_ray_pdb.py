import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
def _connect_ray_pdb(host=None, port=None, patch_stdstreams=False, quiet=None, breakpoint_uuid=None, debugger_external=False):
    """
    Opens a remote PDB on first available port.
    """
    if debugger_external:
        assert not host, 'Cannot specify both host and debugger_external'
        host = '0.0.0.0'
    elif host is None:
        host = os.environ.get('REMOTE_PDB_HOST', '127.0.0.1')
    if port is None:
        port = int(os.environ.get('REMOTE_PDB_PORT', '0'))
    if quiet is None:
        quiet = bool(os.environ.get('REMOTE_PDB_QUIET', ''))
    if not breakpoint_uuid:
        breakpoint_uuid = uuid.uuid4().hex
    if debugger_external:
        ip_address = ray._private.worker.global_worker.node_ip_address
    else:
        ip_address = 'localhost'
    rdb = _RemotePdb(breakpoint_uuid=breakpoint_uuid, host=host, port=port, ip_address=ip_address, patch_stdstreams=patch_stdstreams, quiet=quiet)
    sockname = rdb._listen_socket.getsockname()
    pdb_address = '{}:{}'.format(ip_address, sockname[1])
    parentframeinfo = inspect.getouterframes(inspect.currentframe())[2]
    data = {'proctitle': setproctitle.getproctitle(), 'pdb_address': pdb_address, 'filename': parentframeinfo.filename, 'lineno': parentframeinfo.lineno, 'traceback': '\n'.join(traceback.format_exception(*sys.exc_info())), 'timestamp': time.time(), 'job_id': ray.get_runtime_context().get_job_id()}
    _internal_kv_put('RAY_PDB_{}'.format(breakpoint_uuid), json.dumps(data), overwrite=True, namespace=ray_constants.KV_NAMESPACE_PDB)
    rdb.listen()
    _internal_kv_del('RAY_PDB_{}'.format(breakpoint_uuid), namespace=ray_constants.KV_NAMESPACE_PDB)
    return rdb