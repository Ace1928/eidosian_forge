import hashlib
import hmac
import os
import struct
from ntlm_auth.compute_response import ComputeResponse
from ntlm_auth.constants import AvId, AvFlags, MessageTypes, NegotiateFlags, \
from ntlm_auth.rc4 import ARC4
class NegotiateMessage(object):
    EXPECTED_BODY_LENGTH = 40

    def __init__(self, negotiate_flags, domain_name, workstation):
        """
        [MS-NLMP] v28.0 2016-07-14

        2.2.1.1 NEGOTIATE_MESSAGE
        The NEGOTIATE_MESSAGE defines an NTLM Negotiate message that is sent
        from the client to the server. This message allows the client to
        specify its supported NTLM options to the server.

        :param negotiate_flags: A NEGOTIATE structure that contains a set of
            bit flags. These flags are the options the client supports
        :param domain_name: The domain name of the user to authenticate with,
            default is None
        :param workstation: The worksation of the client machine, default is
            None

        Attributes:
            signature: An 8-byte character array that MUST contain the ASCII
                string 'NTLMSSP\x00'
            message_type: A 32-bit unsigned integer that indicates the message
                type. This field must be set to 0x00000001
            negotiate_flags: A NEGOTIATE structure that contains a set of bit
                flags. These flags are the options the client supports
            version: Contains the windows version info of the client. It is
                used only debugging purposes and are only set when
                NTLMSSP_NEGOTIATE_VERSION flag is set
            domain_name: A byte-array that contains the name of the client
                authentication domain that MUST Be encoded in the negotiated
                character set
            workstation: A byte-array that contains the name of the client
                machine that MUST Be encoded in the negotiated character set
        """
        self.signature = NTLM_SIGNATURE
        self.message_type = struct.pack('<L', MessageTypes.NTLM_NEGOTIATE)
        if domain_name is None:
            self.domain_name = ''
        else:
            self.domain_name = domain_name
            negotiate_flags |= NegotiateFlags.NTLMSSP_NEGOTIATE_OEM_DOMAIN_SUPPLIED
        if workstation is None:
            self.workstation = ''
        else:
            self.workstation = workstation
            negotiate_flags |= NegotiateFlags.NTLMSSP_NEGOTIATE_OEM_WORKSTATION_SUPPLIED
        negotiate_flags |= NegotiateFlags.NTLMSSP_NEGOTIATE_UNICODE
        negotiate_flags &= ~NegotiateFlags.NTLMSSP_NEGOTIATE_OEM
        self.domain_name = self.domain_name.encode('ascii')
        self.workstation = self.workstation.encode('ascii')
        self.version = get_version(negotiate_flags)
        self.negotiate_flags = struct.pack('<I', negotiate_flags)

    def get_data(self):
        payload_offset = self.EXPECTED_BODY_LENGTH
        domain_name_len = struct.pack('<H', len(self.domain_name))
        domain_name_max_len = struct.pack('<H', len(self.domain_name))
        domain_name_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.domain_name)
        workstation_len = struct.pack('<H', len(self.workstation))
        workstation_max_len = struct.pack('<H', len(self.workstation))
        workstation_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.workstation)
        payload = self.domain_name
        payload += self.workstation
        msg1 = self.signature
        msg1 += self.message_type
        msg1 += self.negotiate_flags
        msg1 += domain_name_len
        msg1 += domain_name_max_len
        msg1 += domain_name_buffer_offset
        msg1 += workstation_len
        msg1 += workstation_max_len
        msg1 += workstation_buffer_offset
        msg1 += self.version
        assert self.EXPECTED_BODY_LENGTH == len(msg1), 'BODY_LENGTH: %d != msg1: %d' % (self.EXPECTED_BODY_LENGTH, len(msg1))
        msg1 += payload
        return msg1