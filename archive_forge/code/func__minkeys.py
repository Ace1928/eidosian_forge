import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def _minkeys(self, routelist):
    """Utility function to walk the route backwards

        Will also determine the minimum keys we can handle to generate
        a working route.

        routelist is a list of the '/' split route path
        defaults is a dict of all the defaults provided for the route

        """
    minkeys = []
    backcheck = routelist[:]
    if not self.minimization:
        for part in backcheck:
            if isinstance(part, dict):
                minkeys.append(part['name'])
        return (frozenset(minkeys), backcheck)
    gaps = False
    backcheck.reverse()
    for part in backcheck:
        if not isinstance(part, dict) and part not in self.done_chars:
            gaps = True
            continue
        elif not isinstance(part, dict):
            continue
        key = part['name']
        if key in self.defaults and (not gaps):
            continue
        minkeys.append(key)
        gaps = True
    return (frozenset(minkeys), backcheck)