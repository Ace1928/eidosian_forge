import binascii
import hashlib
import hmac
import struct
import ntlm_auth.compute_keys as compkeys
from ntlm_auth.constants import NegotiateFlags, SignSealConstants
from ntlm_auth.rc4 import ARC4
def _unseal_message(self, message):
    """
        [MS-NLMP] v28.0 2016-07-14

        3.4.3 Message Confidentiality
        Will generate a dencrypted message using RC4 based on the
        ServerSealingKey

        :param message: The message to be unsealed (dencrypted)
        :return decrypted_message: The decrypted message
        """
    decrypted_message = self.incoming_handle.update(message)
    return decrypted_message