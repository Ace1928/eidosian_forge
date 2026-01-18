import struct
class _TZifHeader:
    __slots__ = ['version', 'isutcnt', 'isstdcnt', 'leapcnt', 'timecnt', 'typecnt', 'charcnt']

    def __init__(self, *args):
        for attr, val in zip(self.__slots__, args, strict=True):
            setattr(self, attr, val)

    @classmethod
    def from_file(cls, stream):
        if stream.read(4) != b'TZif':
            raise ValueError('Invalid TZif file: magic not found')
        _version = stream.read(1)
        if _version == b'\x00':
            version = 1
        else:
            version = int(_version)
        stream.read(15)
        args = (version,)
        args = args + struct.unpack('>6l', stream.read(24))
        return cls(*args)