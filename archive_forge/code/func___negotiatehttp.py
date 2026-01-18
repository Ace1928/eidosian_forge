import base64
import socket
import struct
import sys
def __negotiatehttp(self, destaddr, destport):
    """__negotiatehttp(self,destaddr,destport)
        Negotiates a connection through an HTTP server.
        """
    if not self.__proxy[3]:
        addr = socket.gethostbyname(destaddr)
    else:
        addr = destaddr
    headers = ['CONNECT ', addr, ':', str(destport), ' HTTP/1.1\r\n']
    wrote_host_header = False
    wrote_auth_header = False
    if self.__proxy[6] != None:
        for key, val in self.__proxy[6].iteritems():
            headers += [key, ': ', val, '\r\n']
            wrote_host_header = key.lower() == 'host'
            wrote_auth_header = key.lower() == 'proxy-authorization'
    if not wrote_host_header:
        headers += ['Host: ', destaddr, '\r\n']
    if not wrote_auth_header:
        if self.__proxy[4] != None and self.__proxy[5] != None:
            headers += [self.__getauthheader(), '\r\n']
    headers.append('\r\n')
    self.sendall(''.join(headers).encode())
    resp = self.recv(1)
    while resp.find('\r\n\r\n'.encode()) == -1:
        resp = resp + self.recv(1)
    statusline = resp.splitlines()[0].split(' '.encode(), 2)
    if statusline[0] not in ('HTTP/1.0'.encode(), 'HTTP/1.1'.encode()):
        self.close()
        raise GeneralProxyError((1, _generalerrors[1]))
    try:
        statuscode = int(statusline[1])
    except ValueError:
        self.close()
        raise GeneralProxyError((1, _generalerrors[1]))
    if statuscode != 200:
        self.close()
        raise HTTPError((statuscode, statusline[2]))
    self.__proxysockname = ('0.0.0.0', 0)
    self.__proxypeername = (addr, destport)