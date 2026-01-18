import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
class MulticastParser(XMLRecordParser):
    """
    A XMLRecordParser that knows how to process the IANA IPv4 multicast address
    allocation file.

    It can be found online here :-

        - http://www.iana.org/assignments/multicast-addresses/multicast-addresses.xml
    """

    def __init__(self, fh, **kwargs):
        """
        Constructor.

        fh - a valid, open file handle to an IANA IPv4 multicast address
             allocation file.

        kwargs - additional parser options.
        """
        super(MulticastParser, self).__init__(fh)

    def normalise_addr(self, addr):
        """
        Removes variations from address entries found in this particular file.
        """
        if '-' in addr:
            a1, a2 = addr.split('-')
            o1 = a1.strip().split('.')
            o2 = a2.strip().split('.')
            return '%s-%s' % ('.'.join([str(int(i)) for i in o1]), '.'.join([str(int(i)) for i in o2]))
        else:
            o1 = addr.strip().split('.')
            return '.'.join([str(int(i)) for i in o1])

    def process_record(self, rec):
        """
        Callback method invoked for every record.

        See base class method for more details.
        """
        if 'addr' in rec:
            record = {'address': self.normalise_addr(str(rec['addr'])), 'descr': str(rec.get('description', ''))}
            return record