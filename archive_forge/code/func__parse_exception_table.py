import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _parse_exception_table(code):
    iterator = iter(code.co_exceptiontable)
    entries = []
    try:
        while True:
            start = _parse_varint(iterator) * 2
            length = _parse_varint(iterator) * 2
            end = start + length
            target = _parse_varint(iterator) * 2
            dl = _parse_varint(iterator)
            depth = dl >> 1
            lasti = bool(dl & 1)
            entries.append(_ExceptionTableEntry(start, end, target, depth, lasti))
    except StopIteration:
        return entries