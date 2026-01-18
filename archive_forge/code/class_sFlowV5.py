import struct
import logging
@sFlow.register_sflow_version(SFLOW_V5)
class sFlowV5(object):
    _PACK_STR = '!ii'
    _PACK_STR_IPV4 = '!iiIIIII'
    _PACK_STR_IPV6 = '!ii4IIIII'
    _AGENT_IPTYPE_V4 = 1
    _AGENT_IPTYPE_V6 = 2
    _MIN_LEN_V4 = struct.calcsize(_PACK_STR_IPV4)
    _MIN_LEN_V6 = struct.calcsize(_PACK_STR_IPV6)

    def __init__(self, version, address_type, agent_address, sub_agent_id, sequence_number, uptime, samples_num, samples):
        super(sFlowV5, self).__init__()
        self.version = version
        self.address_type = address_type
        self.agent_address = agent_address
        self.sub_agent_id = sub_agent_id
        self.sequence_number = sequence_number
        self.uptime = uptime
        self.samples_num = samples_num
        self.samples = samples

    @classmethod
    def parser(cls, buf):
        version, address_type = struct.unpack_from(cls._PACK_STR, buf)
        if address_type == cls._AGENT_IPTYPE_V4:
            pack_str = cls._PACK_STR_IPV4
            min_len = cls._MIN_LEN_V4
        elif address_type == cls._AGENT_IPTYPE_V6:
            pack_str = cls._PACK_STR_IPV6
            min_len = cls._MIN_LEN_V6
        else:
            LOG.info('Unknown address_type. sFlowV5.address_type=%d', address_type)
            return None
        version, address_type, agent_address, sub_agent_id, sequence_number, uptime, samples_num = struct.unpack_from(pack_str, buf)
        offset = min_len
        samples = []
        while len(buf) > offset:
            sample = sFlowV5Sample.parser(buf, offset)
            offset += sFlowV5Sample.MIN_LEN + sample.sample_length
            samples.append(sample)
        msg = cls(version, address_type, agent_address, sub_agent_id, sequence_number, uptime, samples_num, samples)
        return msg