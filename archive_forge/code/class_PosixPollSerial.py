from __future__ import absolute_import
import errno
import fcntl
import os
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
class PosixPollSerial(Serial):
    """    Poll based read implementation. Not all systems support poll properly.
    However this one has better handling of errors, such as a device
    disconnecting while it's in use (e.g. USB-serial unplugged).
    """

    def read(self, size=1):
        """        Read size bytes from the serial port. If a timeout is set it may
        return less characters as requested. With no timeout it will block
        until the requested number of bytes is read.
        """
        if not self.is_open:
            raise PortNotOpenError()
        read = bytearray()
        timeout = Timeout(self._timeout)
        poll = select.poll()
        poll.register(self.fd, select.POLLIN | select.POLLERR | select.POLLHUP | select.POLLNVAL)
        poll.register(self.pipe_abort_read_r, select.POLLIN | select.POLLERR | select.POLLHUP | select.POLLNVAL)
        if size > 0:
            while len(read) < size:
                for fd, event in poll.poll(None if timeout.is_infinite else timeout.time_left() * 1000):
                    if fd == self.pipe_abort_read_r:
                        break
                    if event & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
                        raise SerialException('device reports error (poll)')
                if fd == self.pipe_abort_read_r:
                    os.read(self.pipe_abort_read_r, 1000)
                    break
                buf = os.read(self.fd, size - len(read))
                read.extend(buf)
                if timeout.expired() or ((self._inter_byte_timeout is not None and self._inter_byte_timeout > 0) and (not buf)):
                    break
        return bytes(read)