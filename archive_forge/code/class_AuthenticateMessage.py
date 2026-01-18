import hashlib
import hmac
import os
import struct
from ntlm_auth.compute_response import ComputeResponse
from ntlm_auth.constants import AvId, AvFlags, MessageTypes, NegotiateFlags, \
from ntlm_auth.rc4 import ARC4
class AuthenticateMessage(object):
    EXPECTED_BODY_LENGTH = 72
    EXPECTED_BODY_LENGTH_WITH_MIC = 88

    def __init__(self, user_name, password, domain_name, workstation, challenge_message, ntlm_compatibility, server_certificate_hash=None, cbt_data=None):
        """
        [MS-NLMP] v28.0 2016-07-14

        2.2.1.3 AUTHENTICATE_MESSAGE
        The AUTHENTICATE_MESSAGE defines an NTLM authenticate message that is
        sent from the client to the server after the CHALLENGE_MESSAGE is
        processed by the client.

        :param user_name: The user name of the user we are trying to
            authenticate with
        :param password: The password of the user we are trying to authenticate
            with
        :param domain_name: The domain name of the user account we are
            authenticated with, default is None
        :param workstation: The workstation we are using to authenticate with,
            default is None
        :param challenge_message: A ChallengeMessage object that was received
            from the server after the negotiate_message
        :param ntlm_compatibility: The Lan Manager Compatibility Level, used to
            determine what NTLM auth version to use, see Ntlm in ntlm.py for
            more details
        :param server_certificate_hash: Deprecated, used cbt_data instead
        :param cbt_data: The GssChannelBindingsStruct that contains the CBT
            data to bind in the auth response

        Message Attributes (Attributes used to compute the message structure):
            signature: An 8-byte character array that MUST contain the ASCII
                string 'NTLMSSP\x00'
            message_type: A 32-bit unsigned integer that indicates the message
                type. This field must be set to 0x00000003
            negotiate_flags: A NEGOTIATE strucutre that contains a set of bit
                flags. These flags are the choices the client has made from the
                CHALLENGE_MESSAGE options
            version: Contains the windows version info of the client. It is
                used only debugging purposes and are only set when
                NTLMSSP_NEGOTIATE_VERSION flag is set
            mic: The message integrity for the NEGOTIATE_MESSAGE,
                CHALLENGE_MESSAGE and AUTHENTICATE_MESSAGE
            lm_challenge_response: An LM_RESPONSE of LMv2_RESPONSE structure
                that contains the computed LM response to the challenge
            nt_challenge_response: An NTLM_RESPONSE or NTLMv2_RESPONSE
                structure that contains the computed NT response to the
                challenge
            domain_name: The domain or computer name hosting the user account,
                MUST be encoded in the negotiated character set
            user_name: The name of the user to be authenticated, MUST be
                encoded in the negotiated character set
            workstation: The name of the computer to which the user is logged
                on, MUST Be encoded in the negotiated character set
            encrypted_random_session_key: The client's encrypted random session
                key

        Non-Message Attributes (Attributes not used in auth message):
            exported_session_key: A randomly generated session key based on
                other keys, used to derive the SIGNKEY and SEALKEY
            target_info: The AV_PAIR structure used in the nt response
                calculation
        """
        self.signature = NTLM_SIGNATURE
        self.message_type = struct.pack('<L', MessageTypes.NTLM_AUTHENTICATE)
        self.negotiate_flags = challenge_message.negotiate_flags
        self.version = get_version(self.negotiate_flags)
        self.mic = None
        if domain_name is None:
            self.domain_name = ''
        else:
            self.domain_name = domain_name
        if workstation is None:
            self.workstation = ''
        else:
            self.workstation = workstation
        if self.negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_UNICODE:
            self.negotiate_flags &= ~NegotiateFlags.NTLMSSP_NEGOTIATE_OEM
            encoding_value = 'utf-16-le'
        else:
            encoding_value = 'ascii'
        self.domain_name = self.domain_name.encode(encoding_value)
        self.user_name = user_name.encode(encoding_value)
        self.workstation = self.workstation.encode(encoding_value)
        compute_response = ComputeResponse(user_name, password, domain_name, challenge_message, ntlm_compatibility)
        self.lm_challenge_response = compute_response.get_lm_challenge_response()
        self.nt_challenge_response, key_exchange_key, target_info = compute_response.get_nt_challenge_response(self.lm_challenge_response, server_certificate_hash, cbt_data)
        self.target_info = target_info
        if self.negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_KEY_EXCH:
            self.exported_session_key = get_random_export_session_key()
            rc4_handle = ARC4(key_exchange_key)
            self.encrypted_random_session_key = rc4_handle.update(self.exported_session_key)
        else:
            self.exported_session_key = key_exchange_key
            self.encrypted_random_session_key = b''
        self.negotiate_flags = struct.pack('<I', self.negotiate_flags)

    def get_data(self):
        if self.mic is None:
            mic = b''
            expected_body_length = self.EXPECTED_BODY_LENGTH
        else:
            mic = self.mic
            expected_body_length = self.EXPECTED_BODY_LENGTH_WITH_MIC
        payload_offset = expected_body_length
        domain_name_len = struct.pack('<H', len(self.domain_name))
        domain_name_max_len = struct.pack('<H', len(self.domain_name))
        domain_name_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.domain_name)
        user_name_len = struct.pack('<H', len(self.user_name))
        user_name_max_len = struct.pack('<H', len(self.user_name))
        user_name_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.user_name)
        workstation_len = struct.pack('<H', len(self.workstation))
        workstation_max_len = struct.pack('<H', len(self.workstation))
        workstation_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.workstation)
        lm_challenge_response_len = struct.pack('<H', len(self.lm_challenge_response))
        lm_challenge_response_max_len = struct.pack('<H', len(self.lm_challenge_response))
        lm_challenge_response_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.lm_challenge_response)
        nt_challenge_response_len = struct.pack('<H', len(self.nt_challenge_response))
        nt_challenge_response_max_len = struct.pack('<H', len(self.nt_challenge_response))
        nt_challenge_response_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.nt_challenge_response)
        encrypted_random_session_key_len = struct.pack('<H', len(self.encrypted_random_session_key))
        encrypted_random_session_key_max_len = struct.pack('<H', len(self.encrypted_random_session_key))
        encrypted_random_session_key_buffer_offset = struct.pack('<I', payload_offset)
        payload_offset += len(self.encrypted_random_session_key)
        payload = self.domain_name
        payload += self.user_name
        payload += self.workstation
        payload += self.lm_challenge_response
        payload += self.nt_challenge_response
        payload += self.encrypted_random_session_key
        msg3 = self.signature
        msg3 += self.message_type
        msg3 += lm_challenge_response_len
        msg3 += lm_challenge_response_max_len
        msg3 += lm_challenge_response_buffer_offset
        msg3 += nt_challenge_response_len
        msg3 += nt_challenge_response_max_len
        msg3 += nt_challenge_response_buffer_offset
        msg3 += domain_name_len
        msg3 += domain_name_max_len
        msg3 += domain_name_buffer_offset
        msg3 += user_name_len
        msg3 += user_name_max_len
        msg3 += user_name_buffer_offset
        msg3 += workstation_len
        msg3 += workstation_max_len
        msg3 += workstation_buffer_offset
        msg3 += encrypted_random_session_key_len
        msg3 += encrypted_random_session_key_max_len
        msg3 += encrypted_random_session_key_buffer_offset
        msg3 += self.negotiate_flags
        msg3 += self.version
        msg3 += mic
        msg3 += payload
        return msg3

    def add_mic(self, negotiate_message, challenge_message):
        if self.target_info is not None:
            av_flags = self.target_info[AvId.MSV_AV_FLAGS]
            if av_flags is not None and av_flags == struct.pack('<L', AvFlags.MIC_PROVIDED):
                self.mic = struct.pack('<IIII', 0, 0, 0, 0)
                negotiate_data = negotiate_message.get_data()
                challenge_data = challenge_message.get_data()
                authenticate_data = self.get_data()
                hmac_data = negotiate_data + challenge_data + authenticate_data
                mic = hmac.new(self.exported_session_key, hmac_data, digestmod=hashlib.md5).digest()
                self.mic = mic