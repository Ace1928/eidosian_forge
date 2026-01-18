import logging
from pyu2f import apdu
from pyu2f import errors
def CmdVersion(self):
    """Obtain the version of the device and test transport format.

    Obtains the version of the device and determines whether to use ISO
    7816-4 or the U2f variant.  This function should be called at least once
    before CmdAuthenticate or CmdRegister to make sure the object is using the
    proper transport for the device.

    Returns:
      The version of the U2F protocol in use.
    """
    self.logger.debug('CmdVersion')
    response = self.InternalSendApdu(apdu.CommandApdu(0, apdu.CMD_VERSION, 0, 0))
    if not response.IsSuccess():
        raise errors.ApduError(response.sw1, response.sw2)
    return response.body