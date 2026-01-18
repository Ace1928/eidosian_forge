import binascii
import hashlib
import hmac
import struct
import ntlm_auth.compute_keys as compkeys
from ntlm_auth.constants import NegotiateFlags, SignSealConstants
from ntlm_auth.rc4 import ARC4
class _NtlmMessageSignature1(object):
    EXPECTED_BODY_LENGTH = 16

    def __init__(self, random_pad, checksum, seq_num):
        """
        [MS-NLMP] v28.0 2016-07-14

        2.2.2.9.1 NTLMSSP_MESSAGE_SIGNATURE
        This version of the NTLMSSP_MESSAGE_SIGNATURE structure MUST be used
        when the NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY flag is not
        negotiated.

        :param random_pad: A 4-byte array that contains the random pad for the
            message
        :param checksum: A 4-byte array that contains the checksum for the
            message
        :param seq_num: A 32-bit unsigned integer that contains the NTLM
            sequence number for this application message
        """
        self.version = b'\x01\x00\x00\x00'
        self.random_pad = random_pad
        self.checksum = checksum
        self.seq_num = seq_num

    def get_data(self):
        signature = self.version
        signature += self.random_pad
        signature += self.checksum
        signature += self.seq_num
        assert self.EXPECTED_BODY_LENGTH == len(signature), 'BODY_LENGTH: %d != signature: %d' % (self.EXPECTED_BODY_LENGTH, len(signature))
        return signature