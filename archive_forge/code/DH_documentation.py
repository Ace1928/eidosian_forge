from Cryptodome.Util.number import long_to_bytes
from Cryptodome.PublicKey.ECC import EccKey
Perform a Diffie-Hellman key agreement.

    Keywords:
      kdf (callable):
        A key derivation function that accepts ``bytes`` as input and returns
        ``bytes``.
      static_priv (EccKey):
        The local static private key. Optional.
      static_pub (EccKey):
        The static public key that belongs to the peer. Optional.
      eph_priv (EccKey):
        The local ephemeral private key, generated for this session. Optional.
      eph_pub (EccKey):
        The ephemeral public key, received from the peer for this session. Optional.

    At least two keys must be passed, of which one is a private key and one
    a public key.

    Returns (bytes):
      The derived secret key material.
    