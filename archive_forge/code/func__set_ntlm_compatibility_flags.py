import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
def _set_ntlm_compatibility_flags(self, ntlm_compatibility):
    if ntlm_compatibility >= 0 and ntlm_compatibility <= 5:
        if ntlm_compatibility == 0:
            self.negotiate_flags |= NegotiateFlags.NTLMSSP_NEGOTIATE_NTLM | NegotiateFlags.NTLMSSP_NEGOTIATE_LM_KEY
        elif ntlm_compatibility == 1:
            self.negotiate_flags |= NegotiateFlags.NTLMSSP_NEGOTIATE_NTLM | NegotiateFlags.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY
        else:
            self.negotiate_flags |= NegotiateFlags.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY
    else:
        raise Exception('Unknown ntlm_compatibility level - expecting value between 0 and 5')