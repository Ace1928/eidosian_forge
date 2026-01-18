from Cryptodome.Signature.pss import MGF1
import Cryptodome.Hash.SHA1
from Cryptodome.Util.py3compat import _copy_bytes
import Cryptodome.Util.number
from Cryptodome.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
from ._pkcs1_oaep_decode import oaep_decode
class PKCS1OAEP_Cipher:
    """Cipher object for PKCS#1 v1.5 OAEP.
    Do not create directly: use :func:`new` instead."""

    def __init__(self, key, hashAlgo, mgfunc, label, randfunc):
        """Initialize this PKCS#1 OAEP cipher object.

        :Parameters:
         key : an RSA key object
                If a private half is given, both encryption and decryption are possible.
                If a public half is given, only encryption is possible.
         hashAlgo : hash object
                The hash function to use. This can be a module under `Cryptodome.Hash`
                or an existing hash object created from any of such modules. If not specified,
                `Cryptodome.Hash.SHA1` is used.
         mgfunc : callable
                A mask generation function that accepts two parameters: a string to
                use as seed, and the lenth of the mask to generate, in bytes.
                If not specified, the standard MGF1 consistent with ``hashAlgo`` is used (a safe choice).
         label : bytes/bytearray/memoryview
                A label to apply to this particular encryption. If not specified,
                an empty string is used. Specifying a label does not improve
                security.
         randfunc : callable
                A function that returns random bytes.

        :attention: Modify the mask generation function only if you know what you are doing.
                    Sender and receiver must use the same one.
        """
        self._key = key
        if hashAlgo:
            self._hashObj = hashAlgo
        else:
            self._hashObj = Cryptodome.Hash.SHA1
        if mgfunc:
            self._mgf = mgfunc
        else:
            self._mgf = lambda x, y: MGF1(x, y, self._hashObj)
        self._label = _copy_bytes(None, None, label)
        self._randfunc = randfunc

    def can_encrypt(self):
        """Legacy function to check if you can call :meth:`encrypt`.

        .. deprecated:: 3.0"""
        return self._key.can_encrypt()

    def can_decrypt(self):
        """Legacy function to check if you can call :meth:`decrypt`.

        .. deprecated:: 3.0"""
        return self._key.can_decrypt()

    def encrypt(self, message):
        """Encrypt a message with PKCS#1 OAEP.

        :param message:
            The message to encrypt, also known as plaintext. It can be of
            variable length, but not longer than the RSA modulus (in bytes)
            minus 2, minus twice the hash output size.
            For instance, if you use RSA 2048 and SHA-256, the longest message
            you can encrypt is 190 byte long.
        :type message: bytes/bytearray/memoryview

        :returns: The ciphertext, as large as the RSA modulus.
        :rtype: bytes

        :raises ValueError:
            if the message is too long.
        """
        modBits = Cryptodome.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        hLen = self._hashObj.digest_size
        mLen = len(message)
        ps_len = k - mLen - 2 * hLen - 2
        if ps_len < 0:
            raise ValueError('Plaintext is too long.')
        lHash = self._hashObj.new(self._label).digest()
        ps = b'\x00' * ps_len
        db = lHash + ps + b'\x01' + _copy_bytes(None, None, message)
        ros = self._randfunc(hLen)
        dbMask = self._mgf(ros, k - hLen - 1)
        maskedDB = strxor(db, dbMask)
        seedMask = self._mgf(maskedDB, hLen)
        maskedSeed = strxor(ros, seedMask)
        em = b'\x00' + maskedSeed + maskedDB
        em_int = bytes_to_long(em)
        m_int = self._key._encrypt(em_int)
        c = long_to_bytes(m_int, k)
        return c

    def decrypt(self, ciphertext):
        """Decrypt a message with PKCS#1 OAEP.

        :param ciphertext: The encrypted message.
        :type ciphertext: bytes/bytearray/memoryview

        :returns: The original message (plaintext).
        :rtype: bytes

        :raises ValueError:
            if the ciphertext has the wrong length, or if decryption
            fails the integrity check (in which case, the decryption
            key is probably wrong).
        :raises TypeError:
            if the RSA key has no private half (i.e. you are trying
            to decrypt using a public key).
        """
        modBits = Cryptodome.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        hLen = self._hashObj.digest_size
        if len(ciphertext) != k or k < hLen + 2:
            raise ValueError('Ciphertext with incorrect length.')
        ct_int = bytes_to_long(ciphertext)
        em = self._key._decrypt_to_bytes(ct_int)
        lHash = self._hashObj.new(self._label).digest()
        maskedSeed = em[1:hLen + 1]
        maskedDB = em[hLen + 1:]
        seedMask = self._mgf(maskedDB, hLen)
        seed = strxor(maskedSeed, seedMask)
        dbMask = self._mgf(seed, k - hLen - 1)
        db = strxor(maskedDB, dbMask)
        res = oaep_decode(em, lHash, db)
        if res <= 0:
            raise ValueError('Incorrect decryption.')
        return db[res:]