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
class MultiAckDetailedGraphWalkerImpl:
    """Graph walker implementation speaking the multi-ack-detailed protocol."""

    def __init__(self, walker) -> None:
        self.walker = walker
        self._common: List[bytes] = []

    def ack(self, have_ref):
        self._common.append(have_ref)
        self.walker.send_ack(have_ref, b'common')

    def next(self):
        while True:
            command, sha = self.walker.read_proto_line(_GRAPH_WALKER_COMMANDS)
            if command is None:
                if self.walker.all_wants_satisfied(self._common):
                    self.walker.send_ack(self._common[-1], b'ready')
                self.walker.send_nak()
                if self.walker.stateless_rpc:
                    return None
            elif command == COMMAND_DONE:
                self.walker.notify_done()
                break
            elif command == COMMAND_HAVE:
                return sha
    __next__ = next

    def handle_done(self, done_required, done_received):
        if done_required and (not done_received):
            return False
        if not done_received and (not self._common):
            return False
        if self._common:
            self.walker.send_ack(self._common[-1])
        else:
            self.walker.send_nak()
        return True