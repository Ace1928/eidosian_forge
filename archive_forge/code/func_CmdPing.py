import logging
from pyu2f import apdu
from pyu2f import errors
def CmdPing(self, data):
    self.logger.debug('CmdPing')
    return self.transport.SendPing(data)