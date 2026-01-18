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
def _step_spnego_token(self, in_token: typing.Optional[bytes]=None, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Optional[bytes]:
    chosen_mech = self._chosen_mech
    context, generated_token = self._context_list[chosen_mech]
    out_token: typing.Optional[bytes] = None
    if not context.complete:
        if generated_token:
            out_token = generated_token
            self._context_list[chosen_mech] = (context, None)
        else:
            out_token = context.step(in_token=in_token, channel_bindings=channel_bindings)
        if self._requires_mech_list_mic:
            self._mic_required = True
    return out_token