from __future__ import unicode_literals
import codecs
from .labels import LABELS
def _iter_encode_generator(input, encode):
    for chunck in input:
        output = encode(chunck)
        if output:
            yield output
    output = encode('', final=True)
    if output:
        yield output