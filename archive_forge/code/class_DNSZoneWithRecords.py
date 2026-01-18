from __future__ import (absolute_import, division, print_function)
class DNSZoneWithRecords(object):

    def __init__(self, zone, records):
        self.zone = zone
        self.records = records

    def __str__(self):
        return '({0}, {1})'.format(self.zone, self.records)

    def __repr__(self):
        return 'DNSZoneWithRecords({0!r}, {1!r})'.format(self.zone, self.records)