from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class InterpolateWrapper:

    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)

    def before_get(self, parser, section, option, value, defaults):
        try:
            return self._original.before_get(parser, section, option, value, defaults)
        except Exception:
            e = sys.exc_info()[1]
            args = list(e.args)
            args[0] = f'Error in file {parser.filename}: {e}'
            e.args = tuple(args)
            e.message = args[0]
            raise