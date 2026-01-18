from twisted.protocols import basic
def _refuseMessage(self, message):
    self.transport.write(message + b'\n')
    self.transport.loseConnection()