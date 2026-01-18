from __future__ import print_function
import contextlib
import datetime
import errno
import logging
import os
import time
import uuid
import sys
import traceback
from systemd import journal, id128
from systemd.journal import _make_line
import pytest
class MockSender:

    def __init__(self):
        self.buf = []

    def send(self, MESSAGE, MESSAGE_ID=None, CODE_FILE=None, CODE_LINE=None, CODE_FUNC=None, **kwargs):
        args = ['MESSAGE=' + MESSAGE]
        if MESSAGE_ID is not None:
            id = getattr(MESSAGE_ID, 'hex', MESSAGE_ID)
            args.append('MESSAGE_ID=' + id)
        if CODE_LINE is CODE_FILE is CODE_FUNC is None:
            CODE_FILE, CODE_LINE, CODE_FUNC = traceback.extract_stack(limit=2)[0][:3]
        if CODE_FILE is not None:
            args.append('CODE_FILE=' + CODE_FILE)
        if CODE_LINE is not None:
            args.append('CODE_LINE={:d}'.format(CODE_LINE))
        if CODE_FUNC is not None:
            args.append('CODE_FUNC=' + CODE_FUNC)
        args.extend((_make_line(key, val) for key, val in kwargs.items()))
        self.buf.append(args)