import asyncio
import socket
import sys
import dns._asyncbackend
import dns._features
import dns.exception
import dns.inet
class DatagramSocket(dns._asyncbackend.DatagramSocket):

    def __init__(self, family, transport, protocol):
        super().__init__(family)
        self.transport = transport
        self.protocol = protocol

    async def sendto(self, what, destination, timeout):
        self.transport.sendto(what, destination)
        return len(what)

    async def recvfrom(self, size, timeout):
        done = _get_running_loop().create_future()
        try:
            assert self.protocol.recvfrom is None
            self.protocol.recvfrom = done
            await _maybe_wait_for(done, timeout)
            return done.result()
        finally:
            self.protocol.recvfrom = None

    async def close(self):
        self.protocol.close()

    async def getpeername(self):
        return self.transport.get_extra_info('peername')

    async def getsockname(self):
        return self.transport.get_extra_info('sockname')

    async def getpeercert(self, timeout):
        raise NotImplementedError