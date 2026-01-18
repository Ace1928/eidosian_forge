import functools
def _reverse_pointer(self):
    """Return the reverse DNS pointer name for the IPv6 address.

        This implements the method described in RFC3596 2.5.

        """
    reverse_chars = self.exploded[::-1].replace(':', '')
    return '.'.join(reverse_chars) + '.ip6.arpa'