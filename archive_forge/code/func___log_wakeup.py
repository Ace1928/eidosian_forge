import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
def __log_wakeup(self, events):
    if not events:
        vlog.dbg('%d-ms timeout' % self.timeout)
    else:
        for fd, revents in events:
            if revents != 0:
                s = ''
                if revents & POLLIN:
                    s += '[POLLIN]'
                if revents & POLLOUT:
                    s += '[POLLOUT]'
                if revents & POLLERR:
                    s += '[POLLERR]'
                if revents & POLLHUP:
                    s += '[POLLHUP]'
                if revents & POLLNVAL:
                    s += '[POLLNVAL]'
                vlog.dbg('%s on fd %d' % (s, fd))