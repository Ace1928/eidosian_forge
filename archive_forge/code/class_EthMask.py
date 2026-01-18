import json
import netaddr
import re
class EthMask(Decoder):
    """EthMask represents an Ethernet address with optional mask.

    It uses netaddr.EUI.

    Attributes:
        eth (netaddr.EUI): The Ethernet address.
        mask (netaddr.EUI): Optional, the Ethernet address mask.

    Args:
        string (str): A string representing the masked Ethernet address
            e.g: 00.11:22:33:44:55 or 01:00:22:00:33:00/01:00:00:00:00:00
    """

    def __init__(self, string):
        mask_parts = string.split('/')
        self._eth = netaddr.EUI(mask_parts[0])
        if len(mask_parts) == 2:
            self._mask = netaddr.EUI(mask_parts[1])
        else:
            self._mask = None

    @property
    def eth(self):
        """The Ethernet address."""
        return self._eth

    @property
    def mask(self):
        """The Ethernet address mask."""
        return self._mask

    def __eq__(self, other):
        """Equality operator.

        Both the Ethernet address and the mask are compared. This can be used
        to implement filters where we expect a specific mask to be present,
        e.g: dl_dst=01:00:00:00:00:00/01:00:00:00:00:00.

        Args:
            other (EthMask): Another EthMask to compare against.

        Returns:
            True if this EthMask is the same as the other.
        """
        return self._mask == other._mask and self._eth == other._eth

    def __contains__(self, other):
        """Contains operator.

        Args:
            other (netaddr.EUI or EthMask): An Ethernet address.

        Returns:
            True if the other netaddr.EUI or fully-masked EthMask is
            contained in this EthMask's address range.
        """
        if isinstance(other, EthMask):
            if other._mask:
                raise ValueError('Comparing non fully-masked EthMask is not supported')
            return other._eth in self
        if self._mask:
            return other.value & self._mask.value == self._eth.value & self._mask.value
        else:
            return other == self._eth

    def __str__(self):
        if self._mask:
            return '/'.join([self._eth.format(netaddr.mac_unix), self._mask.format(netaddr.mac_unix)])
        else:
            return self._eth.format(netaddr.mac_unix)

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self)

    def to_json(self):
        return str(self)