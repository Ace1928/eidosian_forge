import base64
import socket
import struct
import sys
def __negotiatesocks5(self, destaddr, destport):
    """__negotiatesocks5(self,destaddr,destport)
        Negotiates a connection through a SOCKS5 server.
        """
    if self.__proxy[4] != None and self.__proxy[5] != None:
        self.sendall(struct.pack('BBBB', 5, 2, 0, 2))
    else:
        self.sendall(struct.pack('BBB', 5, 1, 0))
    chosenauth = self.__recvall(2)
    if chosenauth[0:1] != chr(5).encode():
        self.close()
        raise GeneralProxyError((1, _generalerrors[1]))
    if chosenauth[1:2] == chr(0).encode():
        pass
    elif chosenauth[1:2] == chr(2).encode():
        self.sendall(chr(1).encode() + chr(len(self.__proxy[4])) + self.__proxy[4] + chr(len(self.__proxy[5])) + self.__proxy[5])
        authstat = self.__recvall(2)
        if authstat[0:1] != chr(1).encode():
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        if authstat[1:2] != chr(0).encode():
            self.close()
            raise Socks5AuthError((3, _socks5autherrors[3]))
    else:
        self.close()
        if chosenauth[1] == chr(255).encode():
            raise Socks5AuthError((2, _socks5autherrors[2]))
        else:
            raise GeneralProxyError((1, _generalerrors[1]))
    req = struct.pack('BBB', 5, 1, 0)
    try:
        ipaddr = socket.inet_aton(destaddr)
        req = req + chr(1).encode() + ipaddr
    except socket.error:
        if self.__proxy[3]:
            ipaddr = None
            req = req + chr(3).encode() + chr(len(destaddr)).encode() + destaddr.encode()
        else:
            ipaddr = socket.inet_aton(socket.gethostbyname(destaddr))
            req = req + chr(1).encode() + ipaddr
    req = req + struct.pack('>H', destport)
    self.sendall(req)
    resp = self.__recvall(4)
    if resp[0:1] != chr(5).encode():
        self.close()
        raise GeneralProxyError((1, _generalerrors[1]))
    elif resp[1:2] != chr(0).encode():
        self.close()
        if ord(resp[1:2]) <= 8:
            raise Socks5Error((ord(resp[1:2]), _socks5errors[ord(resp[1:2])]))
        else:
            raise Socks5Error((9, _socks5errors[9]))
    elif resp[3:4] == chr(1).encode():
        boundaddr = self.__recvall(4)
    elif resp[3:4] == chr(3).encode():
        resp = resp + self.recv(1)
        boundaddr = self.__recvall(ord(resp[4:5]))
    else:
        self.close()
        raise GeneralProxyError((1, _generalerrors[1]))
    boundport = struct.unpack('>H', self.__recvall(2))[0]
    self.__proxysockname = (boundaddr, boundport)
    if ipaddr != None:
        self.__proxypeername = (socket.inet_ntoa(ipaddr), destport)
    else:
        self.__proxypeername = (destaddr, destport)