from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import ResultIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
from axolotl.state.prekeybundle import PreKeyBundle
from axolotl.identitykey import IdentityKey
from axolotl.ecc.curve import Curve
from axolotl.ecc.djbec import DjbECPublicKey
import binascii
import sys
class MissingParametersException(Exception):
    PARAM_KEY = 'key'
    PARAM_IDENTITY = 'identity'
    PARAM_SKEY = 'skey'
    PARAM_REGISTRATION = 'registration'
    __PARAMS = (PARAM_KEY, PARAM_IDENTITY, PARAM_SKEY, PARAM_REGISTRATION)

    def __init__(self, jid, parameters):
        self._jid = jid
        assert type(parameters) in (list, str)
        if type(parameters) is str:
            parameters = list(parameters)
        assert len(parameters) > 0
        for p in parameters:
            assert p in self.__PARAMS, '%s is unrecognized param' % p
        self._parameters = parameters

    @property
    def jid(self):
        return self._jid

    @property
    def parameters(self):
        return self._parameters