import functools
def _explode_shorthand_ip_string(self):
    """Expand a shortened IPv6 address.

        Args:
            ip_str: A string, the IPv6 address.

        Returns:
            A string, the expanded IPv6 address.

        """
    if isinstance(self, IPv6Network):
        ip_str = str(self.network_address)
    elif isinstance(self, IPv6Interface):
        ip_str = str(self.ip)
    else:
        ip_str = str(self)
    ip_int = self._ip_int_from_string(ip_str)
    hex_str = '%032x' % ip_int
    parts = [hex_str[x:x + 4] for x in range(0, 32, 4)]
    if isinstance(self, (_BaseNetwork, IPv6Interface)):
        return '%s/%d' % (':'.join(parts), self._prefixlen)
    return ':'.join(parts)