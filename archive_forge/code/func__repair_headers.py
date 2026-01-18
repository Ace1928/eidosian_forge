import re
import textwrap
import email.message
from ._text import FoldedCase
def _repair_headers(self):

    def redent(value):
        """Correct for RFC822 indentation"""
        if not value or '\n' not in value:
            return value
        return textwrap.dedent(' ' * 8 + value)
    headers = [(key, redent(value)) for key, value in vars(self)['_headers']]
    if self._payload:
        headers.append(('Description', self.get_payload()))
    return headers