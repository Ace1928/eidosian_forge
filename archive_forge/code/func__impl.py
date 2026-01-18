import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def _impl(token_stream):
    tokens = []
    for token in token_stream:
        if isinstance(token, source_class):
            tokens.append(token)
            continue
        if tokens:
            yield _constructor(list(tokens))
            tokens.clear()
        yield token
    if tokens:
        yield _constructor(tokens)