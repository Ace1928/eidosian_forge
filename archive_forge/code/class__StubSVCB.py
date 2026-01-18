from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dns import rdata
from dns.name import Name
from dns.tokenizer import Tokenizer
class _StubSVCB(rdata.Rdata):
    """Stub implementation of SVCB RDATA.

  Wire format support is not needed here, so only trivial storage of the
  presentation format is implemented.
  """

    def __init__(self, rdclass, rdtype, priority, target, params):
        super(_StubSVCB, self).__init__(rdclass, rdtype)
        self._priority = priority
        self._target = target
        self._params = params

    def to_text(self, origin=None, relativize=True, **kwargs):
        tokens = ['%d' % self._priority, self._target.choose_relativity(origin, relativize).to_text()] + self._params
        return ' '.join(tokens)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        priority = tok.get_uint16()
        target = tok.get_name(origin).choose_relativity(origin, relativize)
        params = []
        while True:
            token = tok.get().unescape()
            if token.is_eol_or_eof():
                break
            params.append(token.value)
        return cls(rdclass, rdtype, priority, target, params)