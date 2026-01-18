import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
class TCPGitRequestHandler(socketserver.StreamRequestHandler):

    def __init__(self, handlers, *args, **kwargs) -> None:
        self.handlers = handlers
        socketserver.StreamRequestHandler.__init__(self, *args, **kwargs)

    def handle(self):
        proto = ReceivableProtocol(self.connection.recv, self.wfile.write)
        command, args = proto.read_cmd()
        logger.info('Handling %s request, args=%s', command, args)
        cls = self.handlers.get(command, None)
        if not callable(cls):
            raise GitProtocolError('Invalid service %s' % command)
        h = cls(self.server.backend, args, proto)
        h.handle()