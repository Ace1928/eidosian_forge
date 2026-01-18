import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def _generate_cuid_from_slice(self, _slice, cuid_buffer=None, context=None):
    """
        Pop the slice's call stack, generating a cuid entry whenever a
        `__getattr__` call is encountered.
        """
    call_stack = list(_slice._call_stack)
    rcuid = []
    index = _NotSpecified
    name = None
    while call_stack:
        call_stack_entry = call_stack.pop()
        try:
            call, arg = call_stack_entry
        except ValueError:
            call, arg, kwds = call_stack_entry
        if name is not None:
            if call != IndexedComponent_slice.get_attribute:
                raise ValueError("Cannot create a CUID with a __call__ of anything other than a 'component' attribute")
            if arg != 'component':
                raise ValueError("Cannot create a CUID from a slice with a call to any method other than 'component': got '%s'." % arg)
            arg, name = (name, None)
        if call & (IndexedComponent_slice.SET_MASK | IndexedComponent_slice.DEL_MASK):
            raise ValueError('Cannot create a CUID from a slice that contains `set` or `del` calls: got call %s with argument %s' % (call, arg))
        elif call == IndexedComponent_slice.slice_info:
            comp = arg[0]
            slice_info = arg[1:]
            idx = self._index_from_slice_info(slice_info)
            rcuid.append((comp.local_name, idx))
            parent = comp.parent_block()
            base_cuid = self._generate_cuid(parent, cuid_buffer=cuid_buffer, context=context)
            base_cuid.reverse()
            rcuid.extend(base_cuid)
            assert not call_stack
        elif call == IndexedComponent_slice.get_item:
            if index is not _NotSpecified:
                raise ValueError("Two `get_item` calls, %s and %s, were detected before a`get_attr` call. This is not supported by 'ComponentUID'." % (index, arg))
            index = arg
        elif call == IndexedComponent_slice.call:
            if len(arg) != 1:
                raise ValueError('Cannot create a CUID from a slice with a call that has multiple arguments: got arguments %s.' % (arg,))
            name = arg[0]
            if kwds != {}:
                raise ValueError('Cannot create a CUID from a slice with a call that contains keywords: got keyword dict %s.' % (kwds,))
        elif call == IndexedComponent_slice.get_attribute:
            if index is _NotSpecified:
                index = ()
            elif type(index) is not tuple or len(index) == 1:
                index = (index,)
            rcuid.append((arg, index))
            index = _NotSpecified
    rcuid.reverse()
    return rcuid