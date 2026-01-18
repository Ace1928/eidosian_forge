import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
class dispatcher:
    debug = False
    connected = False
    accepting = False
    connecting = False
    closing = False
    addr = None
    ignore_log_types = frozenset({'warning'})

    def __init__(self, sock=None, map=None):
        if map is None:
            self._map = socket_map
        else:
            self._map = map
        self._fileno = None
        if sock:
            sock.setblocking(False)
            self.set_socket(sock, map)
            self.connected = True
            try:
                self.addr = sock.getpeername()
            except OSError as err:
                if err.errno in (ENOTCONN, EINVAL):
                    self.connected = False
                else:
                    self.del_channel(map)
                    raise
        else:
            self.socket = None

    def __repr__(self):
        status = [self.__class__.__module__ + '.' + self.__class__.__qualname__]
        if self.accepting and self.addr:
            status.append('listening')
        elif self.connected:
            status.append('connected')
        if self.addr is not None:
            try:
                status.append('%s:%d' % self.addr)
            except TypeError:
                status.append(repr(self.addr))
        return '<%s at %#x>' % (' '.join(status), id(self))

    def add_channel(self, map=None):
        if map is None:
            map = self._map
        map[self._fileno] = self

    def del_channel(self, map=None):
        fd = self._fileno
        if map is None:
            map = self._map
        if fd in map:
            del map[fd]
        self._fileno = None

    def create_socket(self, family=socket.AF_INET, type=socket.SOCK_STREAM):
        self.family_and_type = (family, type)
        sock = socket.socket(family, type)
        sock.setblocking(False)
        self.set_socket(sock)

    def set_socket(self, sock, map=None):
        self.socket = sock
        self._fileno = sock.fileno()
        self.add_channel(map)

    def set_reuse_addr(self):
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1)
        except OSError:
            pass

    def readable(self):
        return True

    def writable(self):
        return True

    def listen(self, num):
        self.accepting = True
        if os.name == 'nt' and num > 5:
            num = 5
        return self.socket.listen(num)

    def bind(self, addr):
        self.addr = addr
        return self.socket.bind(addr)

    def connect(self, address):
        self.connected = False
        self.connecting = True
        err = self.socket.connect_ex(address)
        if err in (EINPROGRESS, EALREADY, EWOULDBLOCK) or (err == EINVAL and os.name == 'nt'):
            self.addr = address
            return
        if err in (0, EISCONN):
            self.addr = address
            self.handle_connect_event()
        else:
            raise OSError(err, errorcode[err])

    def accept(self):
        try:
            conn, addr = self.socket.accept()
        except TypeError:
            return None
        except OSError as why:
            if why.errno in (EWOULDBLOCK, ECONNABORTED, EAGAIN):
                return None
            else:
                raise
        else:
            return (conn, addr)

    def send(self, data):
        try:
            result = self.socket.send(data)
            return result
        except OSError as why:
            if why.errno == EWOULDBLOCK:
                return 0
            elif why.errno in _DISCONNECTED:
                self.handle_close()
                return 0
            else:
                raise

    def recv(self, buffer_size):
        try:
            data = self.socket.recv(buffer_size)
            if not data:
                self.handle_close()
                return b''
            else:
                return data
        except OSError as why:
            if why.errno in _DISCONNECTED:
                self.handle_close()
                return b''
            else:
                raise

    def close(self):
        self.connected = False
        self.accepting = False
        self.connecting = False
        self.del_channel()
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError as why:
                if why.errno not in (ENOTCONN, EBADF):
                    raise

    def log(self, message):
        sys.stderr.write('log: %s\n' % str(message))

    def log_info(self, message, type='info'):
        if type not in self.ignore_log_types:
            print('%s: %s' % (type, message))

    def handle_read_event(self):
        if self.accepting:
            self.handle_accept()
        elif not self.connected:
            if self.connecting:
                self.handle_connect_event()
            self.handle_read()
        else:
            self.handle_read()

    def handle_connect_event(self):
        err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if err != 0:
            raise OSError(err, _strerror(err))
        self.handle_connect()
        self.connected = True
        self.connecting = False

    def handle_write_event(self):
        if self.accepting:
            return
        if not self.connected:
            if self.connecting:
                self.handle_connect_event()
        self.handle_write()

    def handle_expt_event(self):
        err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if err != 0:
            self.handle_close()
        else:
            self.handle_expt()

    def handle_error(self):
        nil, t, v, tbinfo = compact_traceback()
        try:
            self_repr = repr(self)
        except:
            self_repr = '<__repr__(self) failed for object at %0x>' % id(self)
        self.log_info('uncaptured python exception, closing channel %s (%s:%s %s)' % (self_repr, t, v, tbinfo), 'error')
        self.handle_close()

    def handle_expt(self):
        self.log_info('unhandled incoming priority event', 'warning')

    def handle_read(self):
        self.log_info('unhandled read event', 'warning')

    def handle_write(self):
        self.log_info('unhandled write event', 'warning')

    def handle_connect(self):
        self.log_info('unhandled connect event', 'warning')

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            self.handle_accepted(*pair)

    def handle_accepted(self, sock, addr):
        sock.close()
        self.log_info('unhandled accepted event', 'warning')

    def handle_close(self):
        self.log_info('unhandled close event', 'warning')
        self.close()