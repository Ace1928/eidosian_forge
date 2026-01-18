import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def _load_properties(self, class_id, buf, offset):
    """Load AMQP properties.

        Given the raw bytes containing the property-flags and property-list
        from a content-frame-header, parse and insert into a dictionary
        stored in this object as an attribute named 'properties'.
        """
    props, offset = PROPERTY_CLASSES[class_id](buf, offset)
    self.properties = props
    return offset