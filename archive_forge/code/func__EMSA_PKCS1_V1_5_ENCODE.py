import Cryptodome.Util.number
from Cryptodome.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Cryptodome.Util.asn1 import DerSequence, DerNull, DerOctetString, DerObjectId
def _EMSA_PKCS1_V1_5_ENCODE(msg_hash, emLen, with_hash_parameters=True):
    """
    Implement the ``EMSA-PKCS1-V1_5-ENCODE`` function, as defined
    in PKCS#1 v2.1 (RFC3447, 9.2).

    ``_EMSA-PKCS1-V1_5-ENCODE`` actually accepts the message ``M`` as input,
    and hash it internally. Here, we expect that the message has already
    been hashed instead.

    :Parameters:
     msg_hash : hash object
            The hash object that holds the digest of the message being signed.
     emLen : int
            The length the final encoding must have, in bytes.
     with_hash_parameters : bool
            If True (default), include NULL parameters for the hash
            algorithm in the ``digestAlgorithm`` SEQUENCE.

    :attention: the early standard (RFC2313) stated that ``DigestInfo``
        had to be BER-encoded. This means that old signatures
        might have length tags in indefinite form, which
        is not supported in DER. Such encoding cannot be
        reproduced by this function.

    :Return: An ``emLen`` byte long string that encodes the hash.
    """
    digestAlgo = DerSequence([DerObjectId(msg_hash.oid).encode()])
    if with_hash_parameters:
        digestAlgo.append(DerNull().encode())
    digest = DerOctetString(msg_hash.digest())
    digestInfo = DerSequence([digestAlgo.encode(), digest.encode()]).encode()
    if emLen < len(digestInfo) + 11:
        raise TypeError('DigestInfo is too long for this RSA key (%d bytes).' % len(digestInfo))
    PS = b'\xff' * (emLen - len(digestInfo) - 3)
    return b'\x00\x01' + PS + b'\x00' + digestInfo