from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
def _parse_incoming_messages(self):
    log.debug('Starting message loop for channel {0}', self)
    try:
        while True:
            self._parse_incoming_message()
    except NoMoreMessages as exc:
        log.debug('Exiting message loop for channel {0}: {1}', self, exc)
        with self:
            err_message = str(exc)
            sent_requests = list(self._sent_requests.values())
            for request in sent_requests:
                response_json = MessageDict(None, {'seq': -1, 'request_seq': request.seq, 'command': request.command, 'success': False, 'message': err_message})
                Response._parse(self, response_json, body=exc)
            assert not len(self._sent_requests)
            self._enqueue_handlers(Disconnect(self), self._handle_disconnect)
            self.close()