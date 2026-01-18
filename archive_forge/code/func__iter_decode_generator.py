from __future__ import unicode_literals
import codecs
from .labels import LABELS
def _iter_decode_generator(input, decoder):
    """Return a generator that first yields the :obj:`Encoding`,
    then yields output chukns as Unicode strings.

    """
    decode = decoder.decode
    input = iter(input)
    for chunck in input:
        output = decode(chunck)
        if output:
            assert decoder.encoding is not None
            yield decoder.encoding
            yield output
            break
    else:
        output = decode(b'', final=True)
        assert decoder.encoding is not None
        yield decoder.encoding
        if output:
            yield output
        return
    for chunck in input:
        output = decode(chunck)
        if output:
            yield output
    output = decode(b'', final=True)
    if output:
        yield output