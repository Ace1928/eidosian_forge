import socket
import warnings
from io import BytesIO
from amqp.serialization import _write_table
class EXTERNAL(SASL):
    """EXTERNAL SASL mechanism.

    Enables external authentication, i.e. not handled through this protocol.
    Only passes 'EXTERNAL' as authentication mechanism, but no further
    authentication data.
    """
    mechanism = b'EXTERNAL'

    def start(self, connection):
        return b''