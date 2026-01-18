from __future__ import (absolute_import, division, print_function)
class DNSRecord(object):

    def __init__(self):
        self.id = None
        self.type = None
        self.prefix = None
        self.target = None
        self.ttl = 86400
        self.extra = {}

    def clone(self):
        result = DNSRecord()
        result.id = self.id
        result.type = self.type
        result.prefix = self.prefix
        result.target = self.target
        result.ttl = self.ttl
        result.extra = dict(self.extra)
        return result

    def __str__(self):
        data = []
        if self.id:
            data.append('id: {0}'.format(self.id))
        data.append('type: {0}'.format(self.type))
        if self.prefix:
            data.append('prefix: "{0}"'.format(self.prefix))
        else:
            data.append('prefix: (none)')
        data.append('target: "{0}"'.format(self.target))
        data.append('ttl: {0}'.format(format_ttl(self.ttl)))
        if self.extra:
            data.append('extra: {0}'.format(self.extra))
        return 'DNSRecord(' + ', '.join(data) + ')'

    def __repr__(self):
        return self.__str__()