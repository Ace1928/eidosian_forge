from binascii import hexlify
from hashlib import md5, sha1

    Compute the digest for the given parameters.

    @param HA1: The H(A1) value, as computed by L{calcHA1}.
    @param HA2: The H(A2) value, as computed by L{calcHA2}.
    @param pszNonce: The challenge nonce.
    @param pszNonceCount: The (client) nonce count value for this response.
    @param pszCNonce: The client nonce.
    @param pszQop: The Quality-of-Protection value.
    