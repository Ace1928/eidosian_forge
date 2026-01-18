import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
class IPv6UnicastParser(XMLRecordParser):
    """
    A XMLRecordParser that understands how to parse and retrieve data records
    from the IANA IPv6 unicast address assignments file.

    It can be found online here :-

        - http://www.iana.org/assignments/ipv6-unicast-address-assignments/ipv6-unicast-address-assignments.xml
    """

    def __init__(self, fh, **kwargs):
        """
        Constructor.

        fh - a valid, open file handle to an IANA IPv6 address space file.

        kwargs - additional parser options.
        """
        super(IPv6UnicastParser, self).__init__(fh)

    def process_record(self, rec):
        """
        Callback method invoked for every record.

        See base class method for more details.
        """
        record = {'status': str(rec.get('status', '')).strip(), 'description': str(rec.get('description', '')).strip(), 'prefix': str(rec.get('prefix', '')).strip(), 'date': str(rec.get('date', '')).strip(), 'whois': str(rec.get('whois', '')).strip()}
        return record