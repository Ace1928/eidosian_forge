import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def _append_message(self, message):
    """Append message to mailbox and return (start, stop) offsets."""
    self._file.seek(0, 2)
    before = self._file.tell()
    if len(self._toc) == 0 and (not self._pending):
        self._pre_mailbox_hook(self._file)
    try:
        self._pre_message_hook(self._file)
        offsets = self._install_message(message)
        self._post_message_hook(self._file)
    except BaseException:
        self._file.truncate(before)
        raise
    self._file.flush()
    self._file_length = self._file.tell()
    return offsets