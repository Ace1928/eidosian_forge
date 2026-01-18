import struct
from pyu2f import errors
def CheckSuccessOrRaise(self):
    if self.sw1 == 105 and self.sw2 == 133:
        raise errors.TUPRequiredError()
    elif self.sw1 == 106 and self.sw2 == 128:
        raise errors.InvalidKeyHandleError()
    elif self.sw1 == 105 and self.sw2 == 132:
        raise errors.InvalidKeyHandleError()
    elif not self.IsSuccess():
        raise errors.ApduError(self.sw1, self.sw2)