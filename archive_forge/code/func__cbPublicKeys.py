import os
from twisted.conch.ssh import agent, channel, keys
from twisted.internet import protocol, reactor
from twisted.logger import Logger
def _cbPublicKeys(self, blobcomm):
    self._log.debug('got {num_keys} public keys', num_keys=len(blobcomm))
    self.blobs = [x[0] for x in blobcomm]