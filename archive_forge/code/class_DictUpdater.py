import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
class DictUpdater(Subscriber):
    """
    Concrete Subscriber that inserts records received from a Publisher into a
    dictionary.
    """

    def __init__(self, dct, topic, unique_key):
        """
        Constructor.

        dct - lookup dict or dict like object to insert records into.

        topic - high-level category name of data to be processed.

        unique_key - key name in data dict that uniquely identifies it.
        """
        self.dct = dct
        self.topic = topic
        self.unique_key = unique_key

    def update(self, data):
        """
        Callback function used by Publisher to notify this Subscriber about
        an update. Stores topic based information into dictionary passed to
        constructor.
        """
        data_id = data[self.unique_key]
        if self.topic == 'IPv4':
            cidr = IPNetwork(cidr_abbrev_to_verbose(data_id))
            self.dct[cidr] = data
        elif self.topic == 'IPv6':
            cidr = IPNetwork(cidr_abbrev_to_verbose(data_id))
            self.dct[cidr] = data
        elif self.topic == 'IPv6_unicast':
            cidr = IPNetwork(data_id)
            self.dct[cidr] = data
        elif self.topic == 'multicast':
            iprange = None
            if '-' in data_id:
                first, last = data_id.split('-')
                iprange = IPRange(first, last)
                cidrs = iprange.cidrs()
                if len(cidrs) == 1:
                    iprange = cidrs[0]
            else:
                iprange = IPAddress(data_id)
            self.dct[iprange] = data