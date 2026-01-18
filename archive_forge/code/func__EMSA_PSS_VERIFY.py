from Cryptodome.Util.py3compat import bchr, bord, iter_range
import Cryptodome.Util.number
from Cryptodome.Util.number import (ceil_div,
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
def _EMSA_PSS_VERIFY(mhash, em, emBits, mgf, sLen):
    """
    Implement the ``EMSA-PSS-VERIFY`` function, as defined
    in PKCS#1 v2.1 (RFC3447, 9.1.2).

    ``EMSA-PSS-VERIFY`` actually accepts the message ``M`` as input,
    and hash it internally. Here, we expect that the message has already
    been hashed instead.

    :Parameters:
      mhash : hash object
        The hash object that holds the digest of the message to be verified.
      em : string
        The signature to verify, therefore proving that the sender really
        signed the message that was received.
      emBits : int
        Length of the final encoding (em), in bits.
      mgf : callable
        A mask generation function that accepts two parameters: a string to
        use as seed, and the lenth of the mask to generate, in bytes.
      sLen : int
        Length of the salt, in bytes.

    :Raise ValueError:
        When the encoding is inconsistent, or the digest or salt lengths
        are too big.
    """
    emLen = ceil_div(emBits, 8)
    lmask = 0
    for i in iter_range(8 * emLen - emBits):
        lmask = lmask >> 1 | 128
    if emLen < mhash.digest_size + sLen + 2:
        raise ValueError('Incorrect signature')
    if ord(em[-1:]) != 188:
        raise ValueError('Incorrect signature')
    maskedDB = em[:emLen - mhash.digest_size - 1]
    h = em[emLen - mhash.digest_size - 1:-1]
    if lmask & bord(em[0]):
        raise ValueError('Incorrect signature')
    dbMask = mgf(h, emLen - mhash.digest_size - 1)
    db = strxor(maskedDB, dbMask)
    db = bchr(bord(db[0]) & ~lmask) + db[1:]
    if not db.startswith(bchr(0) * (emLen - mhash.digest_size - sLen - 2) + bchr(1)):
        raise ValueError('Incorrect signature')
    if sLen > 0:
        salt = db[-sLen:]
    else:
        salt = b''
    m_prime = bchr(0) * 8 + mhash.digest() + salt
    hobj = mhash.new()
    hobj.update(m_prime)
    hp = hobj.digest()
    if h != hp:
        raise ValueError('Incorrect signature')