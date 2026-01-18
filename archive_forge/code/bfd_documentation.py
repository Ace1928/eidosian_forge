import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
Authenticate the SHA1 hash for this packet.

        This method can be invoked only when ``self.auth_hash`` is defined.

        Returns a boolean indicates whether the hash can be authenticated
        by the correspondent Auth Key or not.

        ``prev`` is a ``bfd`` instance for the BFD Control header which this
        authentication section belongs to. It's necessary to be assigned
        because an SHA1 hash must be calculated over the entire BFD Control
        packet.

        ``auth_keys`` is a dictionary of authentication key chain which
        key is an integer of *Auth Key ID* and value is a string of *Auth Key*.
        