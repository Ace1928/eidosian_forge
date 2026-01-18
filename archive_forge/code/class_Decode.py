import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
class Decode:

    def __getattr__(self, key):

        def decode(x):
            if isinstance(x, str):
                return x
            elif not isinstance(x, bytes):
                return decode(str(x))
            else:
                return str(x, encoding=key)
        return decode