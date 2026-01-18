import re
import sys
import types
import copy
import os
import inspect
def _statetoken(s, names):
    nonstate = 1
    parts = s.split('_')
    for i, part in enumerate(parts[1:], 1):
        if part not in names and part != 'ANY':
            break
    if i > 1:
        states = tuple(parts[1:i])
    else:
        states = ('INITIAL',)
    if 'ANY' in states:
        states = tuple(names)
    tokenname = '_'.join(parts[i:])
    return (states, tokenname)