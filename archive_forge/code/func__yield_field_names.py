import collections
import gzip
import io
import logging
import struct
import numpy as np
def _yield_field_names():
    for name, _ in _OLD_HEADER_FORMAT + _NEW_HEADER_FORMAT:
        if not name.startswith('_'):
            yield name
    yield 'raw_vocab'
    yield 'vocab_size'
    yield 'nwords'
    yield 'vectors_ngrams'
    yield 'hidden_output'
    yield 'ntokens'