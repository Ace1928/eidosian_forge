import sys, os
import textwrap
def _match_long_opt(self, opt):
    """_match_long_opt(opt : string) -> string

        Determine which long option string 'opt' matches, ie. which one
        it is an unambiguous abbreviation for.  Raises BadOptionError if
        'opt' doesn't unambiguously match any long option string.
        """
    return _match_abbrev(opt, self._long_opt)