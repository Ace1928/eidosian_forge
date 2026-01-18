import struct
import sys
def _prefix_from_prefix_int(self, prefixlen):
    """Validate and return a prefix length integer.

        Args:
            prefixlen: An integer containing the prefix length.

        Returns:
            The input, possibly converted from long to int.

        Raises:
            NetmaskValueError: If the input is not an integer, or out of range.
        """
    if not isinstance(prefixlen, (int, long)):
        raise NetmaskValueError('%r is not an integer' % prefixlen)
    prefixlen = int(prefixlen)
    if not 0 <= prefixlen <= self._max_prefixlen:
        raise NetmaskValueError('%d is not a valid prefix length' % prefixlen)
    return prefixlen