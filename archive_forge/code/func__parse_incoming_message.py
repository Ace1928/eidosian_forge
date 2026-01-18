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
def _parse_incoming_message(self):
    """Reads incoming messages, parses them, and puts handlers into the queue
        for _run_handlers() to invoke, until the channel is closed.
        """

    def object_hook(d):
        d = MessageDict(None, d)
        if 'seq' in d:
            self._prettify(d)
        d.associate_with = associate_with
        message_dicts.append(d)
        return d

    def associate_with(message):
        for d in message_dicts:
            d.message = message
            del d.associate_with
    message_dicts = []
    decoder = self.stream.json_decoder_factory(object_hook=object_hook)
    message_dict = self.stream.read_json(decoder)
    assert isinstance(message_dict, MessageDict)
    msg_type = message_dict('type', json.enum('event', 'request', 'response'))
    parser = self._message_parsers[msg_type]
    try:
        parser(self, message_dict)
    except InvalidMessageError as exc:
        log.error('Failed to parse message in channel {0}: {1} in:\n{2}', self, str(exc), json.repr(message_dict))
    except Exception as exc:
        if isinstance(exc, NoMoreMessages) and exc.stream is self.stream:
            raise
        log.swallow_exception('Fatal error in channel {0} while parsing:\n{1}', self, json.repr(message_dict))
        os._exit(1)