import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
class BaseSelectorEventLoop(base_events.BaseEventLoop):
    """Selector event loop.

    See events.EventLoop for API specification.
    """

    def __init__(self, selector=None):
        super().__init__()
        if selector is None:
            selector = selectors.DefaultSelector()
        logger.debug('Using selector: %s', selector.__class__.__name__)
        self._selector = selector
        self._make_self_pipe()
        self._transports = weakref.WeakValueDictionary()

    def _make_socket_transport(self, sock, protocol, waiter=None, *, extra=None, server=None):
        return _SelectorSocketTransport(self, sock, protocol, waiter, extra, server)

    def _make_ssl_transport(self, rawsock, protocol, sslcontext, waiter=None, *, server_side=False, server_hostname=None, extra=None, server=None, ssl_handshake_timeout=constants.SSL_HANDSHAKE_TIMEOUT, ssl_shutdown_timeout=constants.SSL_SHUTDOWN_TIMEOUT):
        ssl_protocol = sslproto.SSLProtocol(self, protocol, sslcontext, waiter, server_side, server_hostname, ssl_handshake_timeout=ssl_handshake_timeout, ssl_shutdown_timeout=ssl_shutdown_timeout)
        _SelectorSocketTransport(self, rawsock, ssl_protocol, extra=extra, server=server)
        return ssl_protocol._app_transport

    def _make_datagram_transport(self, sock, protocol, address=None, waiter=None, extra=None):
        return _SelectorDatagramTransport(self, sock, protocol, address, waiter, extra)

    def close(self):
        if self.is_running():
            raise RuntimeError('Cannot close a running event loop')
        if self.is_closed():
            return
        self._close_self_pipe()
        super().close()
        if self._selector is not None:
            self._selector.close()
            self._selector = None

    def _close_self_pipe(self):
        self._remove_reader(self._ssock.fileno())
        self._ssock.close()
        self._ssock = None
        self._csock.close()
        self._csock = None
        self._internal_fds -= 1

    def _make_self_pipe(self):
        self._ssock, self._csock = socket.socketpair()
        self._ssock.setblocking(False)
        self._csock.setblocking(False)
        self._internal_fds += 1
        self._add_reader(self._ssock.fileno(), self._read_from_self)

    def _process_self_data(self, data):
        pass

    def _read_from_self(self):
        while True:
            try:
                data = self._ssock.recv(4096)
                if not data:
                    break
                self._process_self_data(data)
            except InterruptedError:
                continue
            except BlockingIOError:
                break

    def _write_to_self(self):
        csock = self._csock
        if csock is None:
            return
        try:
            csock.send(b'\x00')
        except OSError:
            if self._debug:
                logger.debug('Fail to write a null byte into the self-pipe socket', exc_info=True)

    def _start_serving(self, protocol_factory, sock, sslcontext=None, server=None, backlog=100, ssl_handshake_timeout=constants.SSL_HANDSHAKE_TIMEOUT, ssl_shutdown_timeout=constants.SSL_SHUTDOWN_TIMEOUT):
        self._add_reader(sock.fileno(), self._accept_connection, protocol_factory, sock, sslcontext, server, backlog, ssl_handshake_timeout, ssl_shutdown_timeout)

    def _accept_connection(self, protocol_factory, sock, sslcontext=None, server=None, backlog=100, ssl_handshake_timeout=constants.SSL_HANDSHAKE_TIMEOUT, ssl_shutdown_timeout=constants.SSL_SHUTDOWN_TIMEOUT):
        for _ in range(backlog):
            try:
                conn, addr = sock.accept()
                if self._debug:
                    logger.debug('%r got a new connection from %r: %r', server, addr, conn)
                conn.setblocking(False)
            except (BlockingIOError, InterruptedError, ConnectionAbortedError):
                return None
            except OSError as exc:
                if exc.errno in (errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM):
                    self.call_exception_handler({'message': 'socket.accept() out of system resource', 'exception': exc, 'socket': trsock.TransportSocket(sock)})
                    self._remove_reader(sock.fileno())
                    self.call_later(constants.ACCEPT_RETRY_DELAY, self._start_serving, protocol_factory, sock, sslcontext, server, backlog, ssl_handshake_timeout, ssl_shutdown_timeout)
                else:
                    raise
            else:
                extra = {'peername': addr}
                accept = self._accept_connection2(protocol_factory, conn, extra, sslcontext, server, ssl_handshake_timeout, ssl_shutdown_timeout)
                self.create_task(accept)

    async def _accept_connection2(self, protocol_factory, conn, extra, sslcontext=None, server=None, ssl_handshake_timeout=constants.SSL_HANDSHAKE_TIMEOUT, ssl_shutdown_timeout=constants.SSL_SHUTDOWN_TIMEOUT):
        protocol = None
        transport = None
        try:
            protocol = protocol_factory()
            waiter = self.create_future()
            if sslcontext:
                transport = self._make_ssl_transport(conn, protocol, sslcontext, waiter=waiter, server_side=True, extra=extra, server=server, ssl_handshake_timeout=ssl_handshake_timeout, ssl_shutdown_timeout=ssl_shutdown_timeout)
            else:
                transport = self._make_socket_transport(conn, protocol, waiter=waiter, extra=extra, server=server)
            try:
                await waiter
            except BaseException:
                transport.close()
                waiter = None
                raise
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if self._debug:
                context = {'message': 'Error on transport creation for incoming connection', 'exception': exc}
                if protocol is not None:
                    context['protocol'] = protocol
                if transport is not None:
                    context['transport'] = transport
                self.call_exception_handler(context)

    def _ensure_fd_no_transport(self, fd):
        fileno = fd
        if not isinstance(fileno, int):
            try:
                fileno = int(fileno.fileno())
            except (AttributeError, TypeError, ValueError):
                raise ValueError(f'Invalid file object: {fd!r}') from None
        try:
            transport = self._transports[fileno]
        except KeyError:
            pass
        else:
            if not transport.is_closing():
                raise RuntimeError(f'File descriptor {fd!r} is used by transport {transport!r}')

    def _add_reader(self, fd, callback, *args):
        self._check_closed()
        handle = events.Handle(callback, args, self, None)
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            self._selector.register(fd, selectors.EVENT_READ, (handle, None))
        else:
            mask, (reader, writer) = (key.events, key.data)
            self._selector.modify(fd, mask | selectors.EVENT_READ, (handle, writer))
            if reader is not None:
                reader.cancel()
        return handle

    def _remove_reader(self, fd):
        if self.is_closed():
            return False
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            return False
        else:
            mask, (reader, writer) = (key.events, key.data)
            mask &= ~selectors.EVENT_READ
            if not mask:
                self._selector.unregister(fd)
            else:
                self._selector.modify(fd, mask, (None, writer))
            if reader is not None:
                reader.cancel()
                return True
            else:
                return False

    def _add_writer(self, fd, callback, *args):
        self._check_closed()
        handle = events.Handle(callback, args, self, None)
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            self._selector.register(fd, selectors.EVENT_WRITE, (None, handle))
        else:
            mask, (reader, writer) = (key.events, key.data)
            self._selector.modify(fd, mask | selectors.EVENT_WRITE, (reader, handle))
            if writer is not None:
                writer.cancel()
        return handle

    def _remove_writer(self, fd):
        """Remove a writer callback."""
        if self.is_closed():
            return False
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            return False
        else:
            mask, (reader, writer) = (key.events, key.data)
            mask &= ~selectors.EVENT_WRITE
            if not mask:
                self._selector.unregister(fd)
            else:
                self._selector.modify(fd, mask, (reader, None))
            if writer is not None:
                writer.cancel()
                return True
            else:
                return False

    def add_reader(self, fd, callback, *args):
        """Add a reader callback."""
        self._ensure_fd_no_transport(fd)
        self._add_reader(fd, callback, *args)

    def remove_reader(self, fd):
        """Remove a reader callback."""
        self._ensure_fd_no_transport(fd)
        return self._remove_reader(fd)

    def add_writer(self, fd, callback, *args):
        """Add a writer callback.."""
        self._ensure_fd_no_transport(fd)
        self._add_writer(fd, callback, *args)

    def remove_writer(self, fd):
        """Remove a writer callback."""
        self._ensure_fd_no_transport(fd)
        return self._remove_writer(fd)

    async def sock_recv(self, sock, n):
        """Receive data from the socket.

        The return value is a bytes object representing the data received.
        The maximum amount of data to be received at once is specified by
        nbytes.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        try:
            return sock.recv(n)
        except (BlockingIOError, InterruptedError):
            pass
        fut = self.create_future()
        fd = sock.fileno()
        self._ensure_fd_no_transport(fd)
        handle = self._add_reader(fd, self._sock_recv, fut, sock, n)
        fut.add_done_callback(functools.partial(self._sock_read_done, fd, handle=handle))
        return await fut

    def _sock_read_done(self, fd, fut, handle=None):
        if handle is None or not handle.cancelled():
            self.remove_reader(fd)

    def _sock_recv(self, fut, sock, n):
        if fut.done():
            return
        try:
            data = sock.recv(n)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(data)

    async def sock_recv_into(self, sock, buf):
        """Receive data from the socket.

        The received data is written into *buf* (a writable buffer).
        The return value is the number of bytes written.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        try:
            return sock.recv_into(buf)
        except (BlockingIOError, InterruptedError):
            pass
        fut = self.create_future()
        fd = sock.fileno()
        self._ensure_fd_no_transport(fd)
        handle = self._add_reader(fd, self._sock_recv_into, fut, sock, buf)
        fut.add_done_callback(functools.partial(self._sock_read_done, fd, handle=handle))
        return await fut

    def _sock_recv_into(self, fut, sock, buf):
        if fut.done():
            return
        try:
            nbytes = sock.recv_into(buf)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(nbytes)

    async def sock_recvfrom(self, sock, bufsize):
        """Receive a datagram from a datagram socket.

        The return value is a tuple of (bytes, address) representing the
        datagram received and the address it came from.
        The maximum amount of data to be received at once is specified by
        nbytes.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        try:
            return sock.recvfrom(bufsize)
        except (BlockingIOError, InterruptedError):
            pass
        fut = self.create_future()
        fd = sock.fileno()
        self._ensure_fd_no_transport(fd)
        handle = self._add_reader(fd, self._sock_recvfrom, fut, sock, bufsize)
        fut.add_done_callback(functools.partial(self._sock_read_done, fd, handle=handle))
        return await fut

    def _sock_recvfrom(self, fut, sock, bufsize):
        if fut.done():
            return
        try:
            result = sock.recvfrom(bufsize)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    async def sock_recvfrom_into(self, sock, buf, nbytes=0):
        """Receive data from the socket.

        The received data is written into *buf* (a writable buffer).
        The return value is a tuple of (number of bytes written, address).
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        if not nbytes:
            nbytes = len(buf)
        try:
            return sock.recvfrom_into(buf, nbytes)
        except (BlockingIOError, InterruptedError):
            pass
        fut = self.create_future()
        fd = sock.fileno()
        self._ensure_fd_no_transport(fd)
        handle = self._add_reader(fd, self._sock_recvfrom_into, fut, sock, buf, nbytes)
        fut.add_done_callback(functools.partial(self._sock_read_done, fd, handle=handle))
        return await fut

    def _sock_recvfrom_into(self, fut, sock, buf, bufsize):
        if fut.done():
            return
        try:
            result = sock.recvfrom_into(buf, bufsize)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    async def sock_sendall(self, sock, data):
        """Send data to the socket.

        The socket must be connected to a remote socket. This method continues
        to send data from data until either all data has been sent or an
        error occurs. None is returned on success. On error, an exception is
        raised, and there is no way to determine how much data, if any, was
        successfully processed by the receiving end of the connection.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        try:
            n = sock.send(data)
        except (BlockingIOError, InterruptedError):
            n = 0
        if n == len(data):
            return
        fut = self.create_future()
        fd = sock.fileno()
        self._ensure_fd_no_transport(fd)
        handle = self._add_writer(fd, self._sock_sendall, fut, sock, memoryview(data), [n])
        fut.add_done_callback(functools.partial(self._sock_write_done, fd, handle=handle))
        return await fut

    def _sock_sendall(self, fut, sock, view, pos):
        if fut.done():
            return
        start = pos[0]
        try:
            n = sock.send(view[start:])
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
            return
        start += n
        if start == len(view):
            fut.set_result(None)
        else:
            pos[0] = start

    async def sock_sendto(self, sock, data, address):
        """Send data to the socket.

        The socket must be connected to a remote socket. This method continues
        to send data from data until either all data has been sent or an
        error occurs. None is returned on success. On error, an exception is
        raised, and there is no way to determine how much data, if any, was
        successfully processed by the receiving end of the connection.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        try:
            return sock.sendto(data, address)
        except (BlockingIOError, InterruptedError):
            pass
        fut = self.create_future()
        fd = sock.fileno()
        self._ensure_fd_no_transport(fd)
        handle = self._add_writer(fd, self._sock_sendto, fut, sock, data, address)
        fut.add_done_callback(functools.partial(self._sock_write_done, fd, handle=handle))
        return await fut

    def _sock_sendto(self, fut, sock, data, address):
        if fut.done():
            return
        try:
            n = sock.sendto(data, 0, address)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(n)

    async def sock_connect(self, sock, address):
        """Connect to a remote socket at address.

        This method is a coroutine.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        if sock.family == socket.AF_INET or (base_events._HAS_IPv6 and sock.family == socket.AF_INET6):
            resolved = await self._ensure_resolved(address, family=sock.family, type=sock.type, proto=sock.proto, loop=self)
            _, _, _, _, address = resolved[0]
        fut = self.create_future()
        self._sock_connect(fut, sock, address)
        try:
            return await fut
        finally:
            fut = None

    def _sock_connect(self, fut, sock, address):
        fd = sock.fileno()
        try:
            sock.connect(address)
        except (BlockingIOError, InterruptedError):
            self._ensure_fd_no_transport(fd)
            handle = self._add_writer(fd, self._sock_connect_cb, fut, sock, address)
            fut.add_done_callback(functools.partial(self._sock_write_done, fd, handle=handle))
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(None)
        finally:
            fut = None

    def _sock_write_done(self, fd, fut, handle=None):
        if handle is None or not handle.cancelled():
            self.remove_writer(fd)

    def _sock_connect_cb(self, fut, sock, address):
        if fut.done():
            return
        try:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err != 0:
                raise OSError(err, f'Connect call failed {address}')
        except (BlockingIOError, InterruptedError):
            pass
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(None)
        finally:
            fut = None

    async def sock_accept(self, sock):
        """Accept a connection.

        The socket must be bound to an address and listening for connections.
        The return value is a pair (conn, address) where conn is a new socket
        object usable to send and receive data on the connection, and address
        is the address bound to the socket on the other end of the connection.
        """
        base_events._check_ssl_socket(sock)
        if self._debug and sock.gettimeout() != 0:
            raise ValueError('the socket must be non-blocking')
        fut = self.create_future()
        self._sock_accept(fut, sock)
        return await fut

    def _sock_accept(self, fut, sock):
        fd = sock.fileno()
        try:
            conn, address = sock.accept()
            conn.setblocking(False)
        except (BlockingIOError, InterruptedError):
            self._ensure_fd_no_transport(fd)
            handle = self._add_reader(fd, self._sock_accept, fut, sock)
            fut.add_done_callback(functools.partial(self._sock_read_done, fd, handle=handle))
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result((conn, address))

    async def _sendfile_native(self, transp, file, offset, count):
        del self._transports[transp._sock_fd]
        resume_reading = transp.is_reading()
        transp.pause_reading()
        await transp._make_empty_waiter()
        try:
            return await self.sock_sendfile(transp._sock, file, offset, count, fallback=False)
        finally:
            transp._reset_empty_waiter()
            if resume_reading:
                transp.resume_reading()
            self._transports[transp._sock_fd] = transp

    def _process_events(self, event_list):
        for key, mask in event_list:
            fileobj, (reader, writer) = (key.fileobj, key.data)
            if mask & selectors.EVENT_READ and reader is not None:
                if reader._cancelled:
                    self._remove_reader(fileobj)
                else:
                    self._add_callback(reader)
            if mask & selectors.EVENT_WRITE and writer is not None:
                if writer._cancelled:
                    self._remove_writer(fileobj)
                else:
                    self._add_callback(writer)

    def _stop_serving(self, sock):
        self._remove_reader(sock.fileno())
        sock.close()