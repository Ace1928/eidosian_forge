from __future__ import print_function
from .utilities import key_number_to_key_name
class TimeSignature(object):
    """Container for a Time Signature event, which contains the time signature
    numerator, denominator and the event time in seconds.

    Attributes
    ----------
    numerator : int
        Numerator of time signature.
    denominator : int
        Denominator of time signature.
    time : float
        Time of event in seconds.

    Examples
    --------
    Instantiate a TimeSignature object with 6/8 time signature at 3.14 seconds:

    >>> ts = TimeSignature(6, 8, 3.14)
    >>> print(ts)
    6/8 at 3.14 seconds

    """

    def __init__(self, numerator, denominator, time):
        if not (isinstance(numerator, int) and numerator > 0):
            raise ValueError('{} is not a valid `numerator` type or value'.format(numerator))
        if not (isinstance(denominator, int) and denominator > 0):
            raise ValueError('{} is not a valid `denominator` type or value'.format(denominator))
        if not (isinstance(time, (int, float)) and time >= 0):
            raise ValueError('{} is not a valid `time` type or value'.format(time))
        self.numerator = numerator
        self.denominator = denominator
        self.time = time

    def __repr__(self):
        return 'TimeSignature(numerator={}, denominator={}, time={})'.format(self.numerator, self.denominator, self.time)

    def __str__(self):
        return '{}/{} at {:.2f} seconds'.format(self.numerator, self.denominator, self.time)