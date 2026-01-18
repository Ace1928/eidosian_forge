from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Random import get_random_bytes
def _create_openpgp_cipher(factory, **kwargs):
    """Create a new block cipher, configured in OpenPGP mode.

    :Parameters:
      factory : module
        The module.

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.

      IV : bytes/bytearray/memoryview
        The initialization vector to use for encryption or decryption.

        For encryption, the IV must be as long as the cipher block size.

        For decryption, it must be 2 bytes longer (it is actually the
        *encrypted* IV which was prefixed to the ciphertext).
    """
    iv = kwargs.pop('IV', None)
    IV = kwargs.pop('iv', None)
    if (None, None) == (iv, IV):
        iv = get_random_bytes(factory.block_size)
    if iv is not None:
        if IV is not None:
            raise TypeError("You must either use 'iv' or 'IV', not both")
    else:
        iv = IV
    try:
        key = kwargs.pop('key')
    except KeyError as e:
        raise TypeError('Missing component: ' + str(e))
    return OpenPgpMode(factory, key, iv, kwargs)