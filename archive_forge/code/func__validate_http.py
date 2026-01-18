import copy
import re
import types
from .ucre import build_re
def _validate_http(self, text, pos):
    tail = text[pos:]
    if not self.re.get('http'):
        self.re['http'] = '^\\/\\/' + self.re['src_auth'] + self.re['src_host_port_strict'] + self.re['src_path']
    founds = re.search(self.re['http'], tail, flags=re.IGNORECASE)
    if founds:
        return len(founds.group())
    return 0