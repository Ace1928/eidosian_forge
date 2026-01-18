import struct
import logging
class sFlow(object):
    _PACK_STR = '!i'
    _SFLOW_VERSIONS = {}

    @staticmethod
    def register_sflow_version(version):

        def _register_sflow_version(cls):
            sFlow._SFLOW_VERSIONS[version] = cls
            return cls
        return _register_sflow_version

    def __init__(self):
        super(sFlow, self).__init__()

    @classmethod
    def parser(cls, buf):
        version, = struct.unpack_from(cls._PACK_STR, buf)
        cls_ = cls._SFLOW_VERSIONS.get(version, None)
        if cls_:
            return cls_.parser(buf)
        else:
            return None