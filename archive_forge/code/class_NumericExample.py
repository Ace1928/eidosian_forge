import doctest
import re
import decimal
class NumericExample:
    """
    The actual result (3.141592653589793) differs from the given result,
    but the difference is less than 1e-6, so it still passes::

        >>> 1.5707963267948966 * 2 # doctest: +NUMERIC6
        3.14159285

    The text pieces between the numbers are also compared, performing
    white-space normalization::

        >>> ['a', 4.5, 6.9] # doctest: +NUMERIC6
        ['a', 4.5,       6.9000000001]

    Intervals in the notation emitted by sage are allowed::

        >>> print("4.5?e-1") # doctest: +NUMERIC6
        4.50000000001?e-1

    """