import copy
from collections.abc import Mapping
@staticmethod
def _unpack_items(data):
    if isinstance(data, (dict, Mapping)):
        yield from data.items()
        return
    for i, elem in enumerate(data):
        if len(elem) != 2:
            raise ValueError('dictionary update sequence element #{} has length {}; 2 is required.'.format(i, len(elem)))
        if not isinstance(elem[0], str):
            raise ValueError('Element key %r invalid, only strings are allowed' % elem[0])
        yield elem