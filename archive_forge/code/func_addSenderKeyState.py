from ...state.storageprotos_pb2 import SenderKeyRecordStructure
from .senderkeystate import SenderKeyState
from ...invalidkeyidexception import InvalidKeyIdException
def addSenderKeyState(self, id, iteration, chainKey, signatureKey):
    """
        :type id: int
        :type iteration: int
        :type chainKey: bytearray
        :type signatureKey: ECPublicKey
        """
    self.senderKeyStates.append(SenderKeyState(id, iteration, chainKey, signatureKey))