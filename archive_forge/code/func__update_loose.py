import sys, os
import textwrap
def _update_loose(self, dict):
    """
        Update the option values from an arbitrary dictionary,
        using all keys from the dictionary regardless of whether
        they have a corresponding attribute in self or not.
        """
    self.__dict__.update(dict)