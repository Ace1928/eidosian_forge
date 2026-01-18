from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def _parse_field_descriptor(self, encoding):
    """Parse the 'Field Descriptor' (Metadata) packet.

        This is compatible with MySQL 4.1+ (not compatible with MySQL 4.0).
        """
    self.catalog = self.read_length_coded_string()
    self.db = self.read_length_coded_string()
    self.table_name = self.read_length_coded_string().decode(encoding)
    self.org_table = self.read_length_coded_string().decode(encoding)
    self.name = self.read_length_coded_string().decode(encoding)
    self.org_name = self.read_length_coded_string().decode(encoding)
    self.charsetnr, self.length, self.type_code, self.flags, self.scale = self.read_struct('<xHIBHBxx')