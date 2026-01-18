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
def _step_spnego_input(self, in_token: typing.Optional[bytes]=None, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Tuple[typing.Optional[bytes], typing.Optional[bytes], bool]:
    mech_list_mic = None
    token = None
    is_spnego = True
    if in_token:
        try:
            in_token = unpack_token(in_token)
        except struct.error as e:
            raise InvalidTokenError(base_error=e, context_msg=f'Failed to unpack input token {e!s}')
        if isinstance(in_token, NegTokenInit):
            mech_list_mic = in_token.mech_list_mic
            token = in_token.mech_token
            mech_list = self._rebuild_context_list(mech_types=in_token.mech_types, channel_bindings=channel_bindings)
            if self.usage == 'initiate':
                self._mech_list = mech_list
            else:
                self._init_sent = True
                self._mech_list = in_token.mech_types
                preferred_mech = self._preferred_mech_list()[0]
                if preferred_mech.value != in_token.mech_types[0]:
                    self._mic_required = True
        elif isinstance(in_token, NegTokenResp):
            mech_list_mic = in_token.mech_list_mic
            token = in_token.response_token
            if token and mech_list_mic == token:
                mech_list_mic = None
            if in_token.supported_mech:
                self.__chosen_mech = GSSMech.from_oid(in_token.supported_mech)
                self._mech_sent = True
            if in_token.neg_state == NegState.reject and (not token):
                raise InvalidTokenError(context_msg='Received SPNEGO rejection with no token error message')
            if in_token.neg_state == NegState.request_mic:
                self._mic_required = True
            elif in_token.neg_state == NegState.accept_complete:
                self._complete = True
        else:
            is_spnego = False
            token = in_token
            self.__chosen_mech = GSSMech.ntlm if token and token.startswith(b'NTLMSSP\x00') else GSSMech.kerberos
            if not self._context_list:
                self._rebuild_context_list(mech_types=[self.__chosen_mech.value], channel_bindings=channel_bindings)
    else:
        self._mech_list = self._rebuild_context_list(channel_bindings=channel_bindings)
    return (token, mech_list_mic, is_spnego)