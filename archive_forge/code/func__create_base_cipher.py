import sys
from Cryptodome.Cipher import _create_cipher
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _create_base_cipher(dict_parameters):
    """This method instantiates and returns a smart pointer to
    a low-level base cipher. It will absorb named parameters in
    the process."""
    try:
        key = dict_parameters.pop('key')
    except KeyError:
        raise TypeError("Missing 'key' parameter")
    if len(key) not in key_size:
        raise ValueError('Incorrect Blowfish key length (%d bytes)' % len(key))
    start_operation = _raw_blowfish_lib.Blowfish_start_operation
    stop_operation = _raw_blowfish_lib.Blowfish_stop_operation
    void_p = VoidPointer()
    result = start_operation(c_uint8_ptr(key), c_size_t(len(key)), void_p.address_of())
    if result:
        raise ValueError('Error %X while instantiating the Blowfish cipher' % result)
    return SmartPointer(void_p.get(), stop_operation)