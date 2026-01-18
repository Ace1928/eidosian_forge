import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _enclose(object, alias=''):
    """create a function enclosure around the source of some object"""
    dummy = '__this_is_a_big_dummy_enclosing_function__'
    stub = '__this_is_a_stub_variable__'
    code = 'def %s():\n' % dummy
    code += indent(getsource(object, alias=stub, lstrip=True, force=True))
    code += indent('return %s\n' % stub)
    if alias:
        code += '%s = ' % alias
    code += '%s(); del %s\n' % (dummy, dummy)
    return code