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
def _connect_pdb_client(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    while True:
        read_sockets, write_sockets, error_sockets = select.select([sys.stdin, s], [], [])
        for sock in read_sockets:
            if sock == s:
                data = sock.recv(4096)
                if not data:
                    return
                else:
                    sys.stdout.write(data.decode())
                    sys.stdout.flush()
            else:
                msg = sys.stdin.readline()
                s.send(msg.encode())