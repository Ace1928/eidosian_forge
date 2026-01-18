from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzAes(object):
    """
    Wrapper class for struct `fz_aes`. Not copyable or assignable.
    Structure definitions are public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_aes_crypt_cbc(self, mode, length, iv, input, output):
        """
        Class-aware wrapper for `::fz_aes_crypt_cbc()`.
        	AES block processing. Encrypts or Decrypts (according to mode,
        	which must match what was initially set up) length bytes (which
        	must be a multiple of 16), using (and modifying) the insertion
        	vector iv, reading from input, and writing to output.

        	Never throws an exception.
        """
        return _mupdf.FzAes_fz_aes_crypt_cbc(self, mode, length, iv, input, output)

    def fz_aes_setkey_dec(self, key, keysize):
        """
        Class-aware wrapper for `::fz_aes_setkey_dec()`.
        	AES decryption intialisation. Fills in the supplied context
        	and prepares for decryption using the given key.

        	Returns non-zero for error (key size other than 128/192/256).

        	Never throws an exception.
        """
        return _mupdf.FzAes_fz_aes_setkey_dec(self, key, keysize)

    def fz_aes_setkey_enc(self, key, keysize):
        """
        Class-aware wrapper for `::fz_aes_setkey_enc()`.
        	AES encryption intialisation. Fills in the supplied context
        	and prepares for encryption using the given key.

        	Returns non-zero for error (key size other than 128/192/256).

        	Never throws an exception.
        """
        return _mupdf.FzAes_fz_aes_setkey_enc(self, key, keysize)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_aes`.
        """
        _mupdf.FzAes_swiginit(self, _mupdf.new_FzAes(*args))
    __swig_destroy__ = _mupdf.delete_FzAes

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzAes_m_internal_value(self)
    m_internal = property(_mupdf.FzAes_m_internal_get, _mupdf.FzAes_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzAes_s_num_instances_get, _mupdf.FzAes_s_num_instances_set)