from reportlab.lib.utils import isUnicode
class ArcIV:
    """
	performs 'ArcIV' Stream Encryption of S using key
	Based on what is widely thought to be RSA's ArcIV algorithm.
	It produces output streams that are identical.

	NB there is no separate decoder arciv(arciv(s,key),key) == s
	"""

    def __init__(self, key):
        self._key = key
        self.reset()

    def reset(self):
        """restore the cipher to it's start state"""
        key = self._key
        if isUnicode(key):
            key = key.encode('utf8')
        sbox = list(range(256))
        k = list(range(256))
        lk = len(key)
        for i in sbox:
            k[i] = key[i % lk] % 256
        j = 0
        for i in range(256):
            j = (j + sbox[i] + k[i]) % 256
            sbox[i], sbox[j] = (sbox[j], sbox[i])
        self._sbox, self._i, self._j = (sbox, 0, 0)

    def _encode(self, B):
        """
		return the list of encoded bytes of B, B might be a string or a
		list of integers between 0 <= i <= 255
		"""
        sbox, i, j = (self._sbox, self._i, self._j)
        C = list(B.encode('utf8')) if isinstance(B, str) else list(B) if isinstance(B, bytes) else B[:]
        n = len(C)
        p = 0
        while p < n:
            self._i = i = (i + 1) % 256
            self._j = j = (j + sbox[i]) % 256
            sbox[i], sbox[j] = (sbox[j], sbox[i])
            C[p] = C[p] ^ sbox[(sbox[i] + sbox[j]) % 256]
            p += 1
        return C

    def encode(self, S):
        """ArcIV encode string S"""
        return bytes(self._encode(S))