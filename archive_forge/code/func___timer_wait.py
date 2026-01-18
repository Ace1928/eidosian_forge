import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
def __timer_wait(self, msec):
    if self.timeout < 0 or msec < self.timeout:
        self.timeout = msec