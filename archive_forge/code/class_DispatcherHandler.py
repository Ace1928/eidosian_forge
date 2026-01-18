import asyncore
import binascii
import collections
import errno
import functools
import hashlib
import hmac
import math
import os
import pickle
import socket
import struct
import time
import futurist
from oslo_utils import excutils
from taskflow.engines.action_engine import executor as base
from taskflow import logging
from taskflow import task as ta
from taskflow.types import notifier as nt
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.utils import schema_utils as su
from taskflow.utils import threading_utils
class DispatcherHandler(asyncore.dispatcher):
    """Dispatches from a single connection into a target."""
    CHUNK_SIZE = 8192

    def __init__(self, sock, addr, dispatcher):
        super(DispatcherHandler, self).__init__(map=dispatcher.map, sock=sock)
        self.blobs_to_write = list(dispatcher.challenge_pieces)
        self.reader = Reader(dispatcher.auth_key, self._dispatch)
        self.targets = dispatcher.targets
        self.tied_to = None
        self.challenge_responded = False
        self.ack_pieces = _encode_message(dispatcher.auth_key, ACK, dispatcher.identity, reverse=True)
        self.addr = addr

    def handle_close(self):
        self.close()

    def writable(self):
        return bool(self.blobs_to_write)

    def handle_write(self):
        try:
            blob = self.blobs_to_write.pop()
        except IndexError:
            pass
        else:
            sent = self.send(blob[0:self.CHUNK_SIZE])
            if sent < len(blob):
                self.blobs_to_write.append(blob[sent:])

    def _send_ack(self):
        self.blobs_to_write.extend(self.ack_pieces)

    def _dispatch(self, from_who, msg_decoder_func):
        if not self.challenge_responded:
            msg = msg_decoder_func()
            su.schema_validate(msg, SCHEMAS[CHALLENGE_RESPONSE])
            if msg != CHALLENGE_RESPONSE:
                raise ChallengeIgnored('Discarding connection from %s challenge was not responded to' % self.addr)
            else:
                LOG.trace('Peer %s (%s) has passed challenge sequence', self.addr, from_who)
                self.challenge_responded = True
                self.tied_to = from_who
                self._send_ack()
        else:
            if self.tied_to != from_who:
                raise UnknownSender('Sender %s previously identified as %s changed there identity to %s after challenge sequence' % (self.addr, self.tied_to, from_who))
            try:
                task = self.targets[from_who]
            except KeyError:
                raise UnknownSender('Unknown message from %s (%s) not matched to any known target' % (self.addr, from_who))
            msg = msg_decoder_func()
            su.schema_validate(msg, SCHEMAS[EVENT])
            if LOG.isEnabledFor(logging.TRACE):
                msg_delay = max(0, time.time() - msg['sent_on'])
                LOG.trace('Dispatching message from %s (%s) (it took %0.3f seconds for it to arrive for processing after being sent)', self.addr, from_who, msg_delay)
            task.notifier.notify(msg['event_type'], msg.get('details'))
            self._send_ack()

    def handle_read(self):
        data = self.recv(self.CHUNK_SIZE)
        if len(data) == 0:
            self.handle_close()
        else:
            try:
                self.reader.feed(data)
            except (IOError, UnknownSender):
                LOG.warning('Invalid received message', exc_info=True)
                self.handle_close()
            except (pickle.PickleError, TypeError):
                LOG.warning('Badly formatted message', exc_info=True)
                self.handle_close()
            except (ValueError, su.ValidationError):
                LOG.warning('Failed validating message', exc_info=True)
                self.handle_close()
            except ChallengeIgnored:
                LOG.warning('Failed challenge sequence', exc_info=True)
                self.handle_close()