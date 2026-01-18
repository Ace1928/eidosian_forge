import struct
def _register_netflow_version(cls):
    NetFlow._NETFLOW_VERSIONS[version] = cls
    return cls