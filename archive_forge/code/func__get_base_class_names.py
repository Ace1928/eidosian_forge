import sys
import dis
from typing import List, Tuple, TypeVar
from types import FunctionType
def _get_base_class_names(frame):
    """ Get baseclass names from the code object """
    co, lasti = (frame.f_code, frame.f_lasti)
    code = co.co_code
    extends = []
    add_last_step = False
    for op, oparg in op_stream(code, lasti):
        if op in dis.hasname:
            if not add_last_step:
                extends = []
            if dis.opname[op] == 'LOAD_NAME':
                extends.append(('name', co.co_names[oparg]))
                add_last_step = True
            elif dis.opname[op] == 'LOAD_ATTR':
                extends.append(('attr', co.co_names[oparg]))
                add_last_step = True
            elif dis.opname[op] == 'LOAD_GLOBAL':
                extends.append(('name', co.co_names[oparg]))
                add_last_step = True
            else:
                add_last_step = False
    items = []
    previous_item = []
    for t, s in extends:
        if t == 'name':
            if previous_item:
                items.append(previous_item)
            previous_item = [s]
        else:
            previous_item += [s]
    if previous_item:
        items.append(previous_item)
    return items