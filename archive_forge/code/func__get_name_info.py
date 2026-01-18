import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _get_name_info(name_index, get_name, **extrainfo):
    """Helper to get optional details about named references

       Returns the dereferenced name as both value and repr if the name
       list is defined.
       Otherwise returns the sentinel value dis.UNKNOWN for the value
       and an empty string for its repr.
    """
    if get_name is not None:
        argval = get_name(name_index, **extrainfo)
        return (argval, argval)
    else:
        return (UNKNOWN, '')