import base64
import logging
import struct
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, unify_credentials
from spnego._gss import GSSAPIProxy
from spnego._spnego import (
from spnego._sspi import SSPIProxy
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
def _step_spnego_mic(self, in_mic: typing.Optional[bytes]=None) -> typing.Optional[bytes]:
    if in_mic:
        self.verify(pack_mech_type_list(self._mech_list), in_mic)
        self._reset_ntlm_crypto_state(outgoing=False)
        self._mic_required = True
        self._mic_recv = True
        if self._mic_sent:
            self._complete = True
    if self._context.complete and self._mic_required and (not self._mic_sent):
        out_mic = self.sign(pack_mech_type_list(self._mech_list))
        self._reset_ntlm_crypto_state()
        self._mic_sent = True
        return out_mic
    return None