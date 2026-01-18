import math
import secrets
import uuid as _uu
from typing import List
from typing import Optional
class ShortUUID(object):

    def __init__(self, alphabet: Optional[str]=None) -> None:
        if alphabet is None:
            alphabet = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        self.set_alphabet(alphabet)

    @property
    def _length(self) -> int:
        """Return the necessary length to fit the entire UUID given the current alphabet."""
        return int(math.ceil(math.log(2 ** 128, self._alpha_len)))

    def encode(self, uuid: _uu.UUID, pad_length: Optional[int]=None) -> str:
        """
        Encode a UUID into a string (LSB first) according to the alphabet.

        If leftmost (MSB) bits are 0, the string might be shorter.
        """
        if not isinstance(uuid, _uu.UUID):
            raise ValueError('Input `uuid` must be a UUID object.')
        if pad_length is None:
            pad_length = self._length
        return int_to_string(uuid.int, self._alphabet, padding=pad_length)

    def decode(self, string: str, legacy: bool=False) -> _uu.UUID:
        """
        Decode a string according to the current alphabet into a UUID.

        Raises ValueError when encountering illegal characters or a too-long string.

        If string too short, fills leftmost (MSB) bits with 0.

        Pass `legacy=True` if your UUID was encoded with a ShortUUID version prior to
        1.0.0.
        """
        if not isinstance(string, str):
            raise ValueError('Input `string` must be a str.')
        if legacy:
            string = string[::-1]
        return _uu.UUID(int=string_to_int(string, self._alphabet))

    def uuid(self, name: Optional[str]=None, pad_length: Optional[int]=None) -> str:
        """
        Generate and return a UUID.

        If the name parameter is provided, set the namespace to the provided
        name and generate a UUID.
        """
        if pad_length is None:
            pad_length = self._length
        if name is None:
            u = _uu.uuid4()
        elif name.lower().startswith(('http://', 'https://')):
            u = _uu.uuid5(_uu.NAMESPACE_URL, name)
        else:
            u = _uu.uuid5(_uu.NAMESPACE_DNS, name)
        return self.encode(u, pad_length)

    def random(self, length: Optional[int]=None) -> str:
        """Generate and return a cryptographically secure short random string of `length`."""
        if length is None:
            length = self._length
        return ''.join((secrets.choice(self._alphabet) for _ in range(length)))

    def get_alphabet(self) -> str:
        """Return the current alphabet used for new UUIDs."""
        return ''.join(self._alphabet)

    def set_alphabet(self, alphabet: str) -> None:
        """Set the alphabet to be used for new UUIDs."""
        new_alphabet = list(sorted(set(alphabet)))
        if len(new_alphabet) > 1:
            self._alphabet = new_alphabet
            self._alpha_len = len(self._alphabet)
        else:
            raise ValueError('Alphabet with more than one unique symbols required.')

    def encoded_length(self, num_bytes: int=16) -> int:
        """Return the string length of the shortened UUID."""
        factor = math.log(256) / math.log(self._alpha_len)
        return int(math.ceil(factor * num_bytes))