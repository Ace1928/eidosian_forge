import socket
import warnings
from io import BytesIO
from amqp.serialization import _write_table
class SASL:
    """The base class for all amqp SASL authentication mechanisms.

    You should sub-class this if you're implementing your own authentication.
    """

    @property
    def mechanism(self):
        """Return a bytes containing the SASL mechanism name."""
        raise NotImplementedError

    def start(self, connection):
        """Return the first response to a SASL challenge as a bytes object."""
        raise NotImplementedError