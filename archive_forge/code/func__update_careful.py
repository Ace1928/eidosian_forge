import sys, os
import textwrap
def _update_careful(self, dict):
    """
        Update the option values from an arbitrary dictionary, but only
        use keys from dict that already have a corresponding attribute
        in self.  Any keys in dict without a corresponding attribute
        are silently ignored.
        """
    for attr in dir(self):
        if attr in dict:
            dval = dict[attr]
            if dval is not None:
                setattr(self, attr, dval)