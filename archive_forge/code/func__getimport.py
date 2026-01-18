import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _getimport(head, tail, alias='', verify=True, builtin=False):
    """helper to build a likely import string from head and tail of namespace.
    ('head','tail') are used in the following context: "from head import tail"

    If verify=True, then test the import string before returning it.
    If builtin=True, then force an import for builtins where possible.
    If alias is provided, then rename the object on import.
    """
    if tail in ['Ellipsis', 'NotImplemented'] and head in ['types']:
        head = len.__module__
    elif tail in ['None'] and head in ['types']:
        _alias = '%s = ' % alias if alias else ''
        if alias == tail:
            _alias = ''
        return _alias + '%s\n' % tail
    if head in ['builtins', '__builtin__']:
        if tail == 'ellipsis':
            tail = 'EllipsisType'
        if _intypes(tail):
            head = 'types'
        elif not builtin:
            _alias = '%s = ' % alias if alias else ''
            if alias == tail:
                _alias = ''
            return _alias + '%s\n' % tail
        else:
            pass
    if not head:
        _str = 'import %s' % tail
    else:
        _str = 'from %s import %s' % (head, tail)
    _alias = ' as %s\n' % alias if alias else '\n'
    if alias == tail:
        _alias = '\n'
    _str += _alias
    if verify and (not head.startswith('dill.')):
        try:
            exec(_str)
        except ImportError:
            _head = head.rsplit('.', 1)[0]
            if not _head:
                raise
            if _head != head:
                _str = _getimport(_head, tail, alias, verify)
    return _str