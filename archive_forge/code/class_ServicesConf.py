from typing import Optional
class ServicesConf(SimpleConfFile):
    """
    /etc/services parser

    @ivar services: dict mapping service names to (port, protocol) tuples.
    """
    defaultFilename = '/etc/services'

    def __init__(self):
        self.services = {}

    def parseFields(self, name, portAndProtocol, *aliases):
        try:
            port, protocol = portAndProtocol.split('/')
            port = int(port)
        except BaseException:
            raise InvalidServicesConfError(f'Invalid port/protocol: {repr(portAndProtocol)}')
        self.services[name, protocol] = port
        for alias in aliases:
            self.services[alias, protocol] = port