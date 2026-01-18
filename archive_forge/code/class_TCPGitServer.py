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
class TCPGitServer(socketserver.TCPServer):
    allow_reuse_address = True
    serve = socketserver.TCPServer.serve_forever

    def _make_handler(self, *args, **kwargs):
        return TCPGitRequestHandler(self.handlers, *args, **kwargs)

    def __init__(self, backend, listen_addr, port=TCP_GIT_PORT, handlers=None) -> None:
        self.handlers = dict(DEFAULT_HANDLERS)
        if handlers is not None:
            self.handlers.update(handlers)
        self.backend = backend
        logger.info('Listening for TCP connections on %s:%d', listen_addr, port)
        socketserver.TCPServer.__init__(self, (listen_addr, port), self._make_handler)

    def verify_request(self, request, client_address):
        logger.info('Handling request from %s', client_address)
        return True

    def handle_error(self, request, client_address):
        logger.exception('Exception happened during processing of request from %s', client_address)