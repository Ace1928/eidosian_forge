from ..ecc.curve import Curve
from .bobaxolotlparamaters import BobAxolotlParameters
from .aliceaxolotlparameters import AliceAxolotlParameters
from ..kdf.hkdfv3 import HKDFv3
from ..util.byteutil import ByteUtil
from .rootkey import RootKey
from .chainkey import ChainKey
from ..protocol.ciphertextmessage import CiphertextMessage
class DerivedKeys:

    def __init__(self, rootKey, chainKey):
        """
            :type rootKey: RootKey
            :type  chainKey: ChainKey
            """
        self.rootKey = rootKey
        self.chainKey = chainKey

    def getRootKey(self):
        return self.rootKey

    def getChainKey(self):
        return self.chainKey