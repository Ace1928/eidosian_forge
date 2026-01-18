import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def _genops(data, yield_end_pos=False):
    if isinstance(data, bytes_types):
        data = io.BytesIO(data)
    if hasattr(data, 'tell'):
        getpos = data.tell
    else:
        getpos = lambda: None
    while True:
        pos = getpos()
        code = data.read(1)
        opcode = code2op.get(code.decode('latin-1'))
        if opcode is None:
            if code == b'':
                raise ValueError('pickle exhausted before seeing STOP')
            else:
                raise ValueError('at position %s, opcode %r unknown' % ('<unknown>' if pos is None else pos, code))
        if opcode.arg is None:
            arg = None
        else:
            arg = opcode.arg.reader(data)
        if yield_end_pos:
            yield (opcode, arg, pos, getpos())
        else:
            yield (opcode, arg, pos)
        if code == b'.':
            assert opcode.name == 'STOP'
            break