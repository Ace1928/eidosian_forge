from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
import time, random
from urllib.parse import quote as url_quote
def _split_auth_string(auth_string):
    """ split a digest auth string into individual key=value strings """
    prev = None
    for item in auth_string.split(','):
        try:
            if prev.count('"') == 1:
                prev = '%s,%s' % (prev, item)
                continue
        except AttributeError:
            if prev == None:
                prev = item
                continue
            else:
                return
        yield prev.strip()
        prev = item
    yield prev.strip()