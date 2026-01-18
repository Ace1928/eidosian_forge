import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
class TransportHost(object):
    """A host element of a parsed transport URL."""

    def __init__(self, hostname=None, port=None, username=None, password=None):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password

    def __hash__(self):
        return hash((self.hostname, self.port, self.username, self.password))

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        attrs = []
        for a in ['hostname', 'port', 'username', 'password']:
            v = getattr(self, a)
            if v:
                attrs.append((a, repr(v)))
        values = ', '.join(['%s=%s' % i for i in attrs])
        return '<TransportHost ' + values + '>'