import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
def _main_loop(self):
    while not self._shutdown:
        readfds = [self._requests]
        writefds = []
        deadline = self._scheduler._next_deadline
        pyngus_conn = self._connection and self._connection.pyngus_conn
        if pyngus_conn and self._connection.socket:
            if pyngus_conn.needs_input:
                readfds.append(self._connection)
            if pyngus_conn.has_output:
                writefds.append(self._connection)
            if pyngus_conn.deadline:
                deadline = pyngus_conn.deadline if not deadline else min(deadline, pyngus_conn.deadline)
        if deadline:
            _now = time.monotonic()
            timeout = 0 if deadline <= _now else deadline - _now
        else:
            timeout = None
        try:
            select.select(readfds, writefds, [], timeout)
        except select.error as serror:
            if serror[0] == errno.EINTR:
                LOG.warning('ignoring interrupt from select(): %s', str(serror))
                continue
            raise
        self._requests.process_requests()
        self._connection.read_socket()
        if pyngus_conn and pyngus_conn.deadline:
            _now = time.monotonic()
            if pyngus_conn.deadline <= _now:
                pyngus_conn.process(_now)
        self._connection.write_socket()
        self._scheduler._process()
    LOG.info('eventloop thread exiting, container=%s', self._container.name)