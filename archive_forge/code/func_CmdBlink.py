import logging
from pyu2f import apdu
from pyu2f import errors
def CmdBlink(self, time):
    self.logger.debug('CmdBlink')
    self.transport.SendBlink(time)