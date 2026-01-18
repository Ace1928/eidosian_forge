from .TurboSHAKE128 import TurboSHAKE
Create a new TurboSHAKE256 object.

    Args:
       domain (integer):
         Optional - A domain separation byte, between 0x01 and 0x7F.
         The default value is 0x1F.
       data (bytes/bytearray/memoryview):
        Optional - The very first chunk of the message to hash.
        It is equivalent to an early call to :meth:`update`.

    :Return: A :class:`TurboSHAKE` object
    