import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
class FileTransferBase(protocol.Protocol):
    _log = Logger()
    versions = (3,)
    packetTypes: Dict[int, str] = {}

    def __init__(self):
        self.buf = b''
        self.otherVersion = None

    def sendPacket(self, kind, data):
        self.transport.write(struct.pack('!LB', len(data) + 1, kind) + data)

    def dataReceived(self, data):
        self.buf += data
        while len(self.buf) >= 9:
            header = self.buf[:9]
            length, kind, reqId = struct.unpack('!LBL', header)
            if len(self.buf) < 4 + length:
                return
            data, self.buf = (self.buf[5:4 + length], self.buf[4 + length:])
            packetType = self.packetTypes.get(kind, None)
            if not packetType:
                self._log.info('no packet type for {kind}', kind=kind)
                continue
            f = getattr(self, f'packet_{packetType}', None)
            if not f:
                self._log.info('not implemented: {packetType} data={data!r}', packetType=packetType, data=data[4:])
                self._sendStatus(reqId, FX_OP_UNSUPPORTED, f"don't understand {packetType}")
                continue
            self._log.info('dispatching: {packetType} requestId={reqId}', packetType=packetType, reqId=reqId)
            try:
                f(data)
            except Exception:
                self._log.failure('Failed to handle packet of type {packetType}', packetType=packetType)
                continue

    def _parseAttributes(self, data):
        flags, = struct.unpack('!L', data[:4])
        attrs = {}
        data = data[4:]
        if flags & FILEXFER_ATTR_SIZE == FILEXFER_ATTR_SIZE:
            size, = struct.unpack('!Q', data[:8])
            attrs['size'] = size
            data = data[8:]
        if flags & FILEXFER_ATTR_OWNERGROUP == FILEXFER_ATTR_OWNERGROUP:
            uid, gid = struct.unpack('!2L', data[:8])
            attrs['uid'] = uid
            attrs['gid'] = gid
            data = data[8:]
        if flags & FILEXFER_ATTR_PERMISSIONS == FILEXFER_ATTR_PERMISSIONS:
            perms, = struct.unpack('!L', data[:4])
            attrs['permissions'] = perms
            data = data[4:]
        if flags & FILEXFER_ATTR_ACMODTIME == FILEXFER_ATTR_ACMODTIME:
            atime, mtime = struct.unpack('!2L', data[:8])
            attrs['atime'] = atime
            attrs['mtime'] = mtime
            data = data[8:]
        if flags & FILEXFER_ATTR_EXTENDED == FILEXFER_ATTR_EXTENDED:
            extendedCount, = struct.unpack('!L', data[:4])
            data = data[4:]
            for i in range(extendedCount):
                extendedType, data = getNS(data)
                extendedData, data = getNS(data)
                attrs[f'ext_{nativeString(extendedType)}'] = extendedData
        return (attrs, data)

    def _packAttributes(self, attrs):
        flags = 0
        data = b''
        if 'size' in attrs:
            data += struct.pack('!Q', attrs['size'])
            flags |= FILEXFER_ATTR_SIZE
        if 'uid' in attrs and 'gid' in attrs:
            data += struct.pack('!2L', attrs['uid'], attrs['gid'])
            flags |= FILEXFER_ATTR_OWNERGROUP
        if 'permissions' in attrs:
            data += struct.pack('!L', attrs['permissions'])
            flags |= FILEXFER_ATTR_PERMISSIONS
        if 'atime' in attrs and 'mtime' in attrs:
            data += struct.pack('!2L', attrs['atime'], attrs['mtime'])
            flags |= FILEXFER_ATTR_ACMODTIME
        extended = []
        for k in attrs:
            if k.startswith('ext_'):
                extType = NS(networkString(k[4:]))
                extData = NS(attrs[k])
                extended.append(extType + extData)
        if extended:
            data += struct.pack('!L', len(extended))
            data += b''.join(extended)
            flags |= FILEXFER_ATTR_EXTENDED
        return struct.pack('!L', flags) + data

    def connectionLost(self, reason):
        """
        Called when connection to the remote subsystem was lost.
        """
        super().connectionLost(reason)
        self.connected = False