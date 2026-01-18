import logging
from pyu2f import apdu
from pyu2f import errors
def CmdWink(self):
    self.logger.debug('CmdWink')
    self.transport.SendWink()