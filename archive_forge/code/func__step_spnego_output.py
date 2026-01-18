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
def _step_spnego_output(self, out_token: typing.Optional[bytes]=None, out_mic: typing.Optional[bytes]=None) -> typing.Optional[bytes]:
    final_token: typing.Optional[bytes] = None
    if not self._init_sent:
        self._init_sent = True
        init_kwargs: typing.Dict[str, typing.Any] = {'mech_token': out_token, 'mech_list_mic': out_mic}
        if self.usage == 'accept':
            init_kwargs['hint_name'] = b'not_defined_in_RFC4178@please_ignore'
        final_token = NegTokenInit(self._mech_list, **init_kwargs).pack()
    elif not self.complete:
        state = NegState.accept_incomplete
        supported_mech = None
        if not self._mech_sent:
            supported_mech = self._chosen_mech.value
            if self._mic_required:
                state = NegState.request_mic
            self._mech_sent = True
        if self._context.complete and (not self._mic_required or (self._mic_sent and self._mic_recv)):
            state = NegState.accept_complete
            self._complete = True
        final_token = NegTokenResp(neg_state=state, supported_mech=supported_mech, response_token=out_token, mech_list_mic=out_mic).pack()
    return final_token