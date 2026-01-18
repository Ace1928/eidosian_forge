import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def _c_preprocess_ifdef(csource, want_block_a, definitions={}, rownum=0):
    ending, block_a, defs_a = _c_preprocess_block(csource, definitions=definitions, rownum=rownum)
    if ending == 'else':
        ending, block_b, defs_b = _c_preprocess_block(csource, definitions=definitions, rownum=rownum)
    else:
        block_b = ''
        defs_b = definitions
    assert ending == 'endif'
    if want_block_a:
        return (block_a, defs_a)
    else:
        return (block_b, defs_b)