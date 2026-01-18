import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def _add_positional(self, args):
    return self.addpath(*args)