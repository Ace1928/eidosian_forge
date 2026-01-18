import struct
class NetFlow(object):
    _PACK_STR = '!H'
    _NETFLOW_VERSIONS = {}

    @staticmethod
    def register_netflow_version(version):

        def _register_netflow_version(cls):
            NetFlow._NETFLOW_VERSIONS[version] = cls
            return cls
        return _register_netflow_version

    def __init__(self):
        super(NetFlow, self).__init__()

    @classmethod
    def parser(cls, buf):
        version, = struct.unpack_from(cls._PACK_STR, buf)
        cls_ = cls._NETFLOW_VERSIONS.get(version, None)
        if cls_:
            return cls_.parser(buf)
        else:
            return None