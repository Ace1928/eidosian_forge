import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
def _handle_message(self, msg):
    s = self._new_buffer()
    g = self.clone(s)
    payload = msg._payload
    if isinstance(payload, list):
        g.flatten(msg.get_payload(0), unixfrom=False, linesep=self._NL)
        payload = s.getvalue()
    else:
        payload = self._encode(payload)
    self._fp.write(payload)