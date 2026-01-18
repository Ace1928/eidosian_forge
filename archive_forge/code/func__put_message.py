import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
def _put_message(self, raw_msg):
    self.log.debug('QUEUE - _put_message:')
    msg = t.cast(t.Dict[str, t.Any], jsonapi.loads(raw_msg))
    if msg['type'] == 'event':
        self.log.debug('QUEUE - received event:')
        self.log.debug(msg)
        self.event_callback(msg)
    else:
        self.log.debug('QUEUE - put message:')
        self.log.debug(msg)
        self.message_queue.put_nowait(msg)