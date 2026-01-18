import hashlib
import hmac
def _constant_time_compare(first, second):
    """Return True if both string or binary inputs are equal, otherwise False.

    This function should take a constant amount of time regardless of
    how many characters in the strings match. This function uses an
    approach designed to prevent timing analysis by avoiding
    content-based short circuiting behaviour, making it appropriate
    for cryptography.
    """
    first = str(first)
    second = str(second)
    if len(first) != len(second):
        return False
    result = 0
    for x, y in zip(first, second):
        result |= ord(x) ^ ord(y)
    return result == 0