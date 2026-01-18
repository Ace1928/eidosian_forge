import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
class BytesGenerator(Generator):
    """Generates a bytes version of a Message object tree.

    Functionally identical to the base Generator except that the output is
    bytes and not string.  When surrogates were used in the input to encode
    bytes, these are decoded back to bytes for output.  If the policy has
    cte_type set to 7bit, then the message is transformed such that the
    non-ASCII bytes are properly content transfer encoded, using the charset
    unknown-8bit.

    The outfp object must accept bytes in its write method.
    """

    def write(self, s):
        self._fp.write(s.encode('ascii', 'surrogateescape'))

    def _new_buffer(self):
        return BytesIO()

    def _encode(self, s):
        return s.encode('ascii')

    def _write_headers(self, msg):
        for h, v in msg.raw_items():
            self._fp.write(self.policy.fold_binary(h, v))
        self.write(self._NL)

    def _handle_text(self, msg):
        if msg._payload is None:
            return
        if _has_surrogates(msg._payload) and (not self.policy.cte_type == '7bit'):
            if self._mangle_from_:
                msg._payload = fcre.sub('>From ', msg._payload)
            self._write_lines(msg._payload)
        else:
            super(BytesGenerator, self)._handle_text(msg)
    _writeBody = _handle_text

    @classmethod
    def _compile_re(cls, s, flags):
        return re.compile(s.encode('ascii'), flags)