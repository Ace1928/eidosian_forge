import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
class OUIIndexParser(Publisher):
    """
    A concrete Publisher that parses OUI (Organisationally Unique Identifier)
    records from IEEE text-based registration files

    It notifies registered Subscribers as each record is encountered, passing
    on the record's position relative to the start of the file (offset) and
    the size of the record (in bytes).

    The file processed by this parser is available online from this URL :-

        - http://standards.ieee.org/regauth/oui/oui.txt

    This is a sample of the record structure expected::

        00-CA-FE   (hex)        ACME CORPORATION
        00CAFE     (base 16)        ACME CORPORATION
                        1 MAIN STREET
                        SPRINGFIELD
                        UNITED STATES
    """

    def __init__(self, ieee_file):
        """
        Constructor.

        :param ieee_file: a file-like object or name of file containing OUI
            records. When using a file-like object always open it in binary
            mode otherwise offsets will probably misbehave.
        """
        super(OUIIndexParser, self).__init__()
        if hasattr(ieee_file, 'readline') and hasattr(ieee_file, 'tell'):
            self.fh = ieee_file
        else:
            self.fh = open(ieee_file, 'rb')

    def parse(self):
        """
        Starts the parsing process which detects records and notifies
        registered subscribers as it finds each OUI record.
        """
        skip_header = True
        record = None
        size = 0
        marker = b'(hex)'
        hyphen = b'-'
        empty_string = b''
        while True:
            line = self.fh.readline()
            if not line:
                break
            if skip_header and marker in line:
                skip_header = False
            if skip_header:
                continue
            if marker in line:
                if record is not None:
                    record.append(size)
                    self.notify(record)
                size = len(line)
                offset = self.fh.tell() - len(line)
                oui = line.split()[0]
                index = int(oui.replace(hyphen, empty_string), 16)
                record = [index, offset]
            else:
                size += len(line)
        record.append(size)
        self.notify(record)