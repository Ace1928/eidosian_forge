from __future__ import (absolute_import, division, print_function)
class DNSZone(object):

    def __init__(self, name, info=None):
        self.id = None
        self.name = name
        self.info = info or dict()

    def __str__(self):
        data = []
        if self.id is not None:
            data.append('id: {0}'.format(self.id))
        data.append('name: {0}'.format(self.name))
        data.append('info: {0}'.format(self.info))
        return 'DNSZone(' + ', '.join(data) + ')'

    def __repr__(self):
        return self.__str__()