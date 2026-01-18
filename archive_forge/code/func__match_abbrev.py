import sys, os
import textwrap
def _match_abbrev(s, wordmap):
    """_match_abbrev(s : string, wordmap : {string : Option}) -> string

    Return the string key in 'wordmap' for which 's' is an unambiguous
    abbreviation.  If 's' is found to be ambiguous or doesn't match any of
    'words', raise BadOptionError.
    """
    if s in wordmap:
        return s
    else:
        possibilities = [word for word in wordmap.keys() if word.startswith(s)]
        if len(possibilities) == 1:
            return possibilities[0]
        elif not possibilities:
            raise BadOptionError(s)
        else:
            possibilities.sort()
            raise AmbiguousOptionError(s, possibilities)