from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
class socksocket(_BaseSocket):
    """socksocket([family[, type[, proto]]]) -> socket object

    Open a SOCKS enabled socket. The parameters are the same as
    those of the standard socket init. In order for SOCKS to work,
    you must specify family=AF_INET and proto=0.
    The "type" argument must be either SOCK_STREAM or SOCK_DGRAM.
    """
    default_proxy = None

    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, *args, **kwargs):
        if type not in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            msg = 'Socket type must be stream or datagram, not {!r}'
            raise ValueError(msg.format(type))
        super(socksocket, self).__init__(family, type, proto, *args, **kwargs)
        self._proxyconn = None
        if self.default_proxy:
            self.proxy = self.default_proxy
        else:
            self.proxy = (None, None, None, None, None, None)
        self.proxy_sockname = None
        self.proxy_peername = None
        self._timeout = None

    def _readall(self, file, count):
        """Receive EXACTLY the number of bytes requested from the file object.

        Blocks until the required number of bytes have been received."""
        data = b''
        while len(data) < count:
            d = file.read(count - len(data))
            if not d:
                raise GeneralProxyError('Connection closed unexpectedly')
            data += d
        return data

    def settimeout(self, timeout):
        self._timeout = timeout
        try:
            peer = self.get_proxy_peername()
            super(socksocket, self).settimeout(self._timeout)
        except socket.error:
            pass

    def gettimeout(self):
        return self._timeout

    def setblocking(self, v):
        if v:
            self.settimeout(None)
        else:
            self.settimeout(0.0)

    def set_proxy(self, proxy_type=None, addr=None, port=None, rdns=True, username=None, password=None):
        """ Sets the proxy to be used.

        proxy_type -  The type of the proxy to be used. Three types
                        are supported: PROXY_TYPE_SOCKS4 (including socks4a),
                        PROXY_TYPE_SOCKS5 and PROXY_TYPE_HTTP
        addr -        The address of the server (IP or DNS).
        port -        The port of the server. Defaults to 1080 for SOCKS
                        servers and 8080 for HTTP proxy servers.
        rdns -        Should DNS queries be performed on the remote side
                       (rather than the local side). The default is True.
                       Note: This has no effect with SOCKS4 servers.
        username -    Username to authenticate with to the server.
                       The default is no authentication.
        password -    Password to authenticate with to the server.
                       Only relevant when username is also provided."""
        if hasattr(username, 'encode'):
            username = username.encode()
        if hasattr(password, 'encode'):
            password = password.encode()
        self.proxy = (proxy_type, addr, port, rdns, username if username else None, password if password else None)

    def setproxy(self, *args, **kwargs):
        if 'proxytype' in kwargs:
            kwargs['proxy_type'] = kwargs.pop('proxytype')
        return self.set_proxy(*args, **kwargs)

    def bind(self, *pos, **kw):
        """Implements proxy connection for UDP sockets.

        Happens during the bind() phase."""
        proxy_type, proxy_addr, proxy_port, rdns, username, password = self.proxy
        if not proxy_type or self.type != socket.SOCK_DGRAM:
            return _orig_socket.bind(self, *pos, **kw)
        if self._proxyconn:
            raise socket.error(EINVAL, 'Socket already bound to an address')
        if proxy_type != SOCKS5:
            msg = 'UDP only supported by SOCKS5 proxy type'
            raise socket.error(EOPNOTSUPP, msg)
        super(socksocket, self).bind(*pos, **kw)
        _, port = self.getsockname()
        dst = ('0', port)
        self._proxyconn = _orig_socket()
        proxy = self._proxy_addr()
        self._proxyconn.connect(proxy)
        UDP_ASSOCIATE = b'\x03'
        _, relay = self._SOCKS5_request(self._proxyconn, UDP_ASSOCIATE, dst)
        host, _ = proxy
        _, port = relay
        super(socksocket, self).connect((host, port))
        super(socksocket, self).settimeout(self._timeout)
        self.proxy_sockname = ('0.0.0.0', 0)

    def sendto(self, bytes, *args, **kwargs):
        if self.type != socket.SOCK_DGRAM:
            return super(socksocket, self).sendto(bytes, *args, **kwargs)
        if not self._proxyconn:
            self.bind(('', 0))
        address = args[-1]
        flags = args[:-1]
        header = BytesIO()
        RSV = b'\x00\x00'
        header.write(RSV)
        STANDALONE = b'\x00'
        header.write(STANDALONE)
        self._write_SOCKS5_address(address, header)
        sent = super(socksocket, self).send(header.getvalue() + bytes, *flags, **kwargs)
        return sent - header.tell()

    def send(self, bytes, flags=0, **kwargs):
        if self.type == socket.SOCK_DGRAM:
            return self.sendto(bytes, flags, self.proxy_peername, **kwargs)
        else:
            return super(socksocket, self).send(bytes, flags, **kwargs)

    def recvfrom(self, bufsize, flags=0):
        if self.type != socket.SOCK_DGRAM:
            return super(socksocket, self).recvfrom(bufsize, flags)
        if not self._proxyconn:
            self.bind(('', 0))
        buf = BytesIO(super(socksocket, self).recv(bufsize + 1024, flags))
        buf.seek(2, SEEK_CUR)
        frag = buf.read(1)
        if ord(frag):
            raise NotImplementedError('Received UDP packet fragment')
        fromhost, fromport = self._read_SOCKS5_address(buf)
        if self.proxy_peername:
            peerhost, peerport = self.proxy_peername
            if fromhost != peerhost or peerport not in (0, fromport):
                raise socket.error(EAGAIN, 'Packet filtered')
        return (buf.read(bufsize), (fromhost, fromport))

    def recv(self, *pos, **kw):
        bytes, _ = self.recvfrom(*pos, **kw)
        return bytes

    def close(self):
        if self._proxyconn:
            self._proxyconn.close()
        return super(socksocket, self).close()

    def get_proxy_sockname(self):
        """Returns the bound IP address and port number at the proxy."""
        return self.proxy_sockname
    getproxysockname = get_proxy_sockname

    def get_proxy_peername(self):
        """
        Returns the IP and port number of the proxy.
        """
        return self.getpeername()
    getproxypeername = get_proxy_peername

    def get_peername(self):
        """Returns the IP address and port number of the destination machine.

        Note: get_proxy_peername returns the proxy."""
        return self.proxy_peername
    getpeername = get_peername

    def _negotiate_SOCKS5(self, *dest_addr):
        """Negotiates a stream connection through a SOCKS5 server."""
        CONNECT = b'\x01'
        self.proxy_peername, self.proxy_sockname = self._SOCKS5_request(self, CONNECT, dest_addr)

    def _SOCKS5_request(self, conn, cmd, dst):
        """
        Send SOCKS5 request with given command (CMD field) and
        address (DST field). Returns resolved DST address that was used.
        """
        proxy_type, addr, port, rdns, username, password = self.proxy
        writer = conn.makefile('wb')
        reader = conn.makefile('rb', 0)
        try:
            if username and password:
                writer.write(b'\x05\x02\x00\x02')
            else:
                writer.write(b'\x05\x01\x00')
            writer.flush()
            chosen_auth = self._readall(reader, 2)
            if chosen_auth[0:1] != b'\x05':
                raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
            if chosen_auth[1:2] == b'\x02':
                writer.write(b'\x01' + chr(len(username)).encode() + username + chr(len(password)).encode() + password)
                writer.flush()
                auth_status = self._readall(reader, 2)
                if auth_status[0:1] != b'\x01':
                    raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
                if auth_status[1:2] != b'\x00':
                    raise SOCKS5AuthError('SOCKS5 authentication failed')
            elif chosen_auth[1:2] != b'\x00':
                if chosen_auth[1:2] == b'\xff':
                    raise SOCKS5AuthError('All offered SOCKS5 authentication methods were rejected')
                else:
                    raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
            writer.write(b'\x05' + cmd + b'\x00')
            resolved = self._write_SOCKS5_address(dst, writer)
            writer.flush()
            resp = self._readall(reader, 3)
            if resp[0:1] != b'\x05':
                raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
            status = ord(resp[1:2])
            if status != 0:
                error = SOCKS5_ERRORS.get(status, 'Unknown error')
                raise SOCKS5Error('{0:#04x}: {1}'.format(status, error))
            bnd = self._read_SOCKS5_address(reader)
            super(socksocket, self).settimeout(self._timeout)
            return (resolved, bnd)
        finally:
            reader.close()
            writer.close()

    def _write_SOCKS5_address(self, addr, file):
        """
        Return the host and port packed for the SOCKS5 protocol,
        and the resolved address as a tuple object.
        """
        host, port = addr
        proxy_type, _, _, rdns, username, password = self.proxy
        family_to_byte = {socket.AF_INET: b'\x01', socket.AF_INET6: b'\x04'}
        for family in (socket.AF_INET, socket.AF_INET6):
            try:
                addr_bytes = socket.inet_pton(family, host)
                file.write(family_to_byte[family] + addr_bytes)
                host = socket.inet_ntop(family, addr_bytes)
                file.write(struct.pack('>H', port))
                return (host, port)
            except socket.error:
                continue
        if rdns:
            host_bytes = host.encode('idna')
            file.write(b'\x03' + chr(len(host_bytes)).encode() + host_bytes)
        else:
            addresses = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP, socket.AI_ADDRCONFIG)
            target_addr = addresses[0]
            family = target_addr[0]
            host = target_addr[4][0]
            addr_bytes = socket.inet_pton(family, host)
            file.write(family_to_byte[family] + addr_bytes)
            host = socket.inet_ntop(family, addr_bytes)
        file.write(struct.pack('>H', port))
        return (host, port)

    def _read_SOCKS5_address(self, file):
        atyp = self._readall(file, 1)
        if atyp == b'\x01':
            addr = socket.inet_ntoa(self._readall(file, 4))
        elif atyp == b'\x03':
            length = self._readall(file, 1)
            addr = self._readall(file, ord(length))
        elif atyp == b'\x04':
            addr = socket.inet_ntop(socket.AF_INET6, self._readall(file, 16))
        else:
            raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
        port = struct.unpack('>H', self._readall(file, 2))[0]
        return (addr, port)

    def _negotiate_SOCKS4(self, dest_addr, dest_port):
        """Negotiates a connection through a SOCKS4 server."""
        proxy_type, addr, port, rdns, username, password = self.proxy
        writer = self.makefile('wb')
        reader = self.makefile('rb', 0)
        try:
            remote_resolve = False
            try:
                addr_bytes = socket.inet_aton(dest_addr)
            except socket.error:
                if rdns:
                    addr_bytes = b'\x00\x00\x00\x01'
                    remote_resolve = True
                else:
                    addr_bytes = socket.inet_aton(socket.gethostbyname(dest_addr))
            writer.write(struct.pack('>BBH', 4, 1, dest_port))
            writer.write(addr_bytes)
            if username:
                writer.write(username)
            writer.write(b'\x00')
            if remote_resolve:
                writer.write(dest_addr.encode('idna') + b'\x00')
            writer.flush()
            resp = self._readall(reader, 8)
            if resp[0:1] != b'\x00':
                raise GeneralProxyError('SOCKS4 proxy server sent invalid data')
            status = ord(resp[1:2])
            if status != 90:
                error = SOCKS4_ERRORS.get(status, 'Unknown error')
                raise SOCKS4Error('{0:#04x}: {1}'.format(status, error))
            self.proxy_sockname = (socket.inet_ntoa(resp[4:]), struct.unpack('>H', resp[2:4])[0])
            if remote_resolve:
                self.proxy_peername = (socket.inet_ntoa(addr_bytes), dest_port)
            else:
                self.proxy_peername = (dest_addr, dest_port)
        finally:
            reader.close()
            writer.close()

    def _negotiate_HTTP(self, dest_addr, dest_port):
        """Negotiates a connection through an HTTP server.

        NOTE: This currently only supports HTTP CONNECT-style proxies."""
        proxy_type, addr, port, rdns, username, password = self.proxy
        addr = dest_addr if rdns else socket.gethostbyname(dest_addr)
        http_headers = [b'CONNECT ' + addr.encode('idna') + b':' + str(dest_port).encode() + b' HTTP/1.1', b'Host: ' + dest_addr.encode('idna')]
        if username and password:
            http_headers.append(b'Proxy-Authorization: basic ' + b64encode(username + b':' + password))
        http_headers.append(b'\r\n')
        self.sendall(b'\r\n'.join(http_headers))
        fobj = self.makefile()
        status_line = fobj.readline()
        fobj.close()
        if not status_line:
            raise GeneralProxyError('Connection closed unexpectedly')
        try:
            proto, status_code, status_msg = status_line.split(' ', 2)
        except ValueError:
            raise GeneralProxyError('HTTP proxy server sent invalid response')
        if not proto.startswith('HTTP/'):
            raise GeneralProxyError('Proxy server does not appear to be an HTTP proxy')
        try:
            status_code = int(status_code)
        except ValueError:
            raise HTTPError('HTTP proxy server did not return a valid HTTP status')
        if status_code != 200:
            error = '{0}: {1}'.format(status_code, status_msg)
            if status_code in (400, 403, 405):
                error += '\n[*] Note: The HTTP proxy server may not be supported by PySocks (must be a CONNECT tunnel proxy)'
            raise HTTPError(error)
        self.proxy_sockname = (b'0.0.0.0', 0)
        self.proxy_peername = (addr, dest_port)
    _proxy_negotiators = {SOCKS4: _negotiate_SOCKS4, SOCKS5: _negotiate_SOCKS5, HTTP: _negotiate_HTTP}

    @set_self_blocking
    def connect(self, dest_pair):
        """
        Connects to the specified destination through a proxy.
        Uses the same API as socket's connect().
        To select the proxy server, use set_proxy().

        dest_pair - 2-tuple of (IP/hostname, port).
        """
        if len(dest_pair) != 2 or dest_pair[0].startswith('['):
            raise socket.error("PySocks doesn't support IPv6: %s" % str(dest_pair))
        dest_addr, dest_port = dest_pair
        if self.type == socket.SOCK_DGRAM:
            if not self._proxyconn:
                self.bind(('', 0))
            dest_addr = socket.gethostbyname(dest_addr)
            if dest_addr == '0.0.0.0' and (not dest_port):
                self.proxy_peername = None
            else:
                self.proxy_peername = (dest_addr, dest_port)
            return
        proxy_type, proxy_addr, proxy_port, rdns, username, password = self.proxy
        if not isinstance(dest_pair, (list, tuple)) or len(dest_pair) != 2 or (not dest_addr) or (not isinstance(dest_port, int)):
            raise GeneralProxyError('Invalid destination-connection (host, port) pair')
        super(socksocket, self).settimeout(self._timeout)
        if proxy_type is None:
            self.proxy_peername = dest_pair
            super(socksocket, self).settimeout(self._timeout)
            super(socksocket, self).connect((dest_addr, dest_port))
            return
        proxy_addr = self._proxy_addr()
        try:
            super(socksocket, self).connect(proxy_addr)
        except socket.error as error:
            self.close()
            proxy_addr, proxy_port = proxy_addr
            proxy_server = '{0}:{1}'.format(proxy_addr, proxy_port)
            printable_type = PRINTABLE_PROXY_TYPES[proxy_type]
            msg = 'Error connecting to {0} proxy {1}'.format(printable_type, proxy_server)
            log.debug('%s due to: %s', msg, error)
            raise ProxyConnectionError(msg, error)
        else:
            try:
                negotiate = self._proxy_negotiators[proxy_type]
                negotiate(self, dest_addr, dest_port)
            except socket.error as error:
                self.close()
                raise GeneralProxyError('Socket error', error)
            except ProxyError:
                self.close()
                raise

    def _proxy_addr(self):
        """
        Return proxy address to connect to as tuple object
        """
        proxy_type, proxy_addr, proxy_port, rdns, username, password = self.proxy
        proxy_port = proxy_port or DEFAULT_PORTS.get(proxy_type)
        if not proxy_port:
            raise GeneralProxyError('Invalid proxy type')
        return (proxy_addr, proxy_port)