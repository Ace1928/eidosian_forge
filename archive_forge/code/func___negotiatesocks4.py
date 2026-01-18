import base64
import socket
import struct
import sys
def __negotiatesocks4(self, destaddr, destport):
    """__negotiatesocks4(self,destaddr,destport)
        Negotiates a connection through a SOCKS4 server.
        """
    rmtrslv = False
    try:
        ipaddr = socket.inet_aton(destaddr)
    except socket.error:
        if self.__proxy[3]:
            ipaddr = struct.pack('BBBB', 0, 0, 0, 1)
            rmtrslv = True
        else:
            ipaddr = socket.inet_aton(socket.gethostbyname(destaddr))
    req = struct.pack('>BBH', 4, 1, destport) + ipaddr
    if self.__proxy[4] != None:
        req = req + self.__proxy[4]
    req = req + chr(0).encode()
    if rmtrslv:
        req = req + destaddr + chr(0).encode()
    self.sendall(req)
    resp = self.__recvall(8)
    if resp[0:1] != chr(0).encode():
        self.close()
        raise GeneralProxyError((1, _generalerrors[1]))
    if resp[1:2] != chr(90).encode():
        self.close()
        if ord(resp[1:2]) in (91, 92, 93):
            self.close()
            raise Socks4Error((ord(resp[1:2]), _socks4errors[ord(resp[1:2]) - 90]))
        else:
            raise Socks4Error((94, _socks4errors[4]))
    self.__proxysockname = (socket.inet_ntoa(resp[4:]), struct.unpack('>H', resp[2:4])[0])
    if rmtrslv != None:
        self.__proxypeername = (socket.inet_ntoa(ipaddr), destport)
    else:
        self.__proxypeername = (destaddr, destport)