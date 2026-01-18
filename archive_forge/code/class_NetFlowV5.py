import struct
@NetFlow.register_netflow_version(NETFLOW_V5)
class NetFlowV5(object):
    _PACK_STR = '!HHIIIIBBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, version, count, sys_uptime, unix_secs, unix_nsecs, flow_sequence, engine_type, engine_id, sampling_interval, flows=None):
        self.version = version
        self.count = count
        self.sys_uptime = sys_uptime
        self.unix_secs = unix_secs
        self.unix_nsecs = unix_nsecs
        self.flow_sequence = flow_sequence
        self.engine_type = engine_type
        self.engine_id = engine_id
        self.sampling_interval = sampling_interval

    @classmethod
    def parser(cls, buf):
        version, count, sys_uptime, unix_secs, unix_nsecs, flow_sequence, engine_type, engine_id, sampling_interval = struct.unpack_from(cls._PACK_STR, buf)
        msg = cls(version, count, sys_uptime, unix_secs, unix_nsecs, flow_sequence, engine_type, engine_id, sampling_interval)
        offset = cls._MIN_LEN
        msg.flows = []
        while len(buf) > offset:
            f = NetFlowV5Flow.parser(buf, offset)
            offset += NetFlowV5Flow._MIN_LEN
            msg.flows.append(f)
        return msg