import abc
import collections
import collections.abc
import operator
import sys
import typing
def _no_slots_copy(dct):
    dict_copy = dict(dct)
    if '__slots__' in dict_copy:
        for slot in dict_copy['__slots__']:
            dict_copy.pop(slot, None)
    return dict_copy