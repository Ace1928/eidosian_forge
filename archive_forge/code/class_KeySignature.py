from __future__ import print_function
from .utilities import key_number_to_key_name
class KeySignature(object):
    """Contains the key signature and the event time in seconds.
    Only supports major and minor keys.

    Attributes
    ----------
    key_number : int
        Key number according to ``[0, 11]`` Major, ``[12, 23]`` minor.
        For example, 0 is C Major, 12 is C minor.
    time : float
        Time of event in seconds.

    Examples
    --------
    Instantiate a C# minor KeySignature object at 3.14 seconds:

    >>> ks = KeySignature(13, 3.14)
    >>> print(ks)
    C# minor at 3.14 seconds
    """

    def __init__(self, key_number, time):
        if not all((isinstance(key_number, int), key_number >= 0, key_number < 24)):
            raise ValueError('{} is not a valid `key_number` type or value'.format(key_number))
        if not (isinstance(time, (int, float)) and time >= 0):
            raise ValueError('{} is not a valid `time` type or value'.format(time))
        self.key_number = key_number
        self.time = time

    def __repr__(self):
        return 'KeySignature(key_number={}, time={})'.format(self.key_number, self.time)

    def __str__(self):
        return '{} at {:.2f} seconds'.format(key_number_to_key_name(self.key_number), self.time)