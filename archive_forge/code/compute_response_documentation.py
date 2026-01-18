import base64
import calendar
import hashlib
import hmac
import os
import struct
import time
import ntlm_auth.compute_hash as comphash
import ntlm_auth.compute_keys as compkeys
import ntlm_auth.messages
from ntlm_auth.des import DES
from ntlm_auth.constants import AvId, AvFlags, NegotiateFlags
from ntlm_auth.gss_channel_bindings import GssChannelBindingsStruct

            [MS-NLMP] v28.0 page 45 - 2016-07-14

            3.1.5.12 Client Received a CHALLENGE_MESSAGE from the Server
            If NTLMv2 authentication is used and the CHALLENGE_MESSAGE
            TargetInfo field has an MsvAvTimestamp present, the client SHOULD
            NOT send the LmChallengeResponse and SHOULD send Z(24) instead.
            