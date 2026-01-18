from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _create_ecb_cipher(factory, **kwargs):
    """Instantiate a cipher object that performs ECB encryption/decryption.

    :Parameters:
      factory : module
        The underlying block cipher, a module from ``Cryptodome.Cipher``.

    All keywords are passed to the underlying block cipher.
    See the relevant documentation for details (at least ``key`` will need
    to be present"""
    cipher_state = factory._create_base_cipher(kwargs)
    cipher_state.block_size = factory.block_size
    if kwargs:
        raise TypeError('Unknown parameters for ECB: %s' % str(kwargs))
    return EcbMode(cipher_state)