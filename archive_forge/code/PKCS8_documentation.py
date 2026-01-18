from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (
from Cryptodome.IO._PBES import PBES1, PBES2, PbesError
Unwrap a private key from a PKCS#8 blob (clear or encrypted).

    Args:
      p8_private_key (bytes):
        The private key wrapped into a PKCS#8 container, DER encoded.

    Keyword Args:
      passphrase (byte string or string):
        The passphrase to use to decrypt the blob (if it is encrypted).

    Return:
      A tuple containing

       #. the algorithm identifier of the wrapped key (OID, dotted string)
       #. the private key (bytes, DER encoded)
       #. the associated parameters (bytes, DER encoded) or ``None``

    Raises:
      ValueError : if decoding fails
    