import os
import collections.abc
Determine if a given string is a dpkg wildcard [debarch_is_wildcard]

        This method is the closest match to dpkg's Dpkg::Arch::debarch_is_wildcard function.

        >>> arch_table = DpkgArchTable.load_arch_table()
        >>> arch_table.is_wildcard("linux-any")
        True
        >>> arch_table.is_wildcard("amd64")
        False
        >>> arch_table.is_wildcard("unknown")
        False
        >>> # Compatibility with the dpkg version of the function.
        >>> arch_table.is_wildcard("unknown-any")
        True

        Compatibility note: The original dpkg function does not ensure that the wildcard matches
          any supported architecture and this re-implementation matches that behaviour.  Therefore,
          this method can return True for a wildcard that can never match anything in practice.

        :param wildcard: A string that might represent a dpkg architecture or wildcard.
        :returns: True the parameter is a known dpkg wildcard.
        