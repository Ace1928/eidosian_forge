import struct
from pyu2f import errors
Serialize the command in the legacy format.

    Encodes the command as per the U2F specs, using the legacy
    encoding in which LC is always present.

    Returns:
      Python bytearray of the encoded command.
    