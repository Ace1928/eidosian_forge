from libcloud.common.base import XmlResponse, ConnectionKey
class ZonomiException(Exception):

    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.args = (code, message)

    def __str__(self):
        return '{} {}'.format(self.code, self.message)

    def __repr__(self):
        return 'ZonomiException {} {}'.format(self.code, self.message)