import hashlib
import logging
from scrapy.utils.misc import create_instance
def _path_safe(text):
    """
    Return a filesystem-safe version of a string ``text``

    >>> _path_safe('simple.org').startswith('simple.org')
    True
    >>> _path_safe('dash-underscore_.org').startswith('dash-underscore_.org')
    True
    >>> _path_safe('some@symbol?').startswith('some_symbol_')
    True
    """
    pathable_slot = ''.join([c if c.isalnum() or c in '-._' else '_' for c in text])
    unique_slot = hashlib.md5(text.encode('utf8')).hexdigest()
    return '-'.join([pathable_slot, unique_slot])