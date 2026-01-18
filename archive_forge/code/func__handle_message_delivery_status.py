import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
def _handle_message_delivery_status(self, msg):
    blocks = []
    for part in msg.get_payload():
        s = self._new_buffer()
        g = self.clone(s)
        g.flatten(part, unixfrom=False, linesep=self._NL)
        text = s.getvalue()
        lines = text.split(self._encoded_NL)
        if lines and lines[-1] == self._encoded_EMPTY:
            blocks.append(self._encoded_NL.join(lines[:-1]))
        else:
            blocks.append(text)
    self._fp.write(self._encoded_NL.join(blocks))