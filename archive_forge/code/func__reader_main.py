import datetime
import enum
import logging
import socket
import sys
import threading
import msgpack
from oslo_privsep._i18n import _
from oslo_utils import uuidutils
def _reader_main(self, reader):
    """This thread owns and demuxes the read channel"""
    with self.lock:
        self.running = True
    for msg in reader:
        msgid, data = msg
        if msgid is None:
            self.out_of_band(data)
        else:
            with self.lock:
                if msgid not in self.outstanding_msgs:
                    LOG.warning('msgid should be in oustanding_msgs, it ispossible that timeout is reached!')
                    continue
                self.outstanding_msgs[msgid].set_result(data)
    LOG.debug('EOF on privsep read channel')
    exc = IOError(_('Premature eof waiting for privileged process'))
    with self.lock:
        for mbox in self.outstanding_msgs.values():
            mbox.set_exception(exc)
        self.running = False