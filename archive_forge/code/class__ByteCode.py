from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
class _ByteCode(object):
    """
    The decoded bytecode of a function, and related information.
    """
    __slots__ = ('func_id', 'co_names', 'co_varnames', 'co_consts', 'co_cellvars', 'co_freevars', 'exception_entries', 'table', 'labels')

    def __init__(self, func_id):
        code = func_id.code
        labels = set((x + _FIXED_OFFSET for x in dis.findlabels(code.co_code)))
        labels.add(0)
        table = OrderedDict(ByteCodeIter(code))
        self._compute_lineno(table, code)
        self.func_id = func_id
        self.co_names = code.co_names
        self.co_varnames = code.co_varnames
        self.co_consts = code.co_consts
        self.co_cellvars = code.co_cellvars
        self.co_freevars = code.co_freevars
        self.table = table
        self.labels = sorted(labels)

    @classmethod
    def _compute_lineno(cls, table, code):
        """
        Compute the line numbers for all bytecode instructions.
        """
        for offset, lineno in dis.findlinestarts(code):
            adj_offset = offset + _FIXED_OFFSET
            if adj_offset in table:
                table[adj_offset].lineno = lineno
        known = code.co_firstlineno
        for inst in table.values():
            if inst.lineno >= 0:
                known = inst.lineno
            else:
                inst.lineno = known
        return table

    def __iter__(self):
        return iter(self.table.values())

    def __getitem__(self, offset):
        return self.table[offset]

    def __contains__(self, offset):
        return offset in self.table

    def dump(self):

        def label_marker(i):
            if i[1].offset in self.labels:
                return '>'
            else:
                return ' '
        return '\n'.join(('%s %10s\t%s' % ((label_marker(i),) + i) for i in self.table.items() if i[1].opname != 'CACHE'))

    @classmethod
    def _compute_used_globals(cls, func, table, co_consts, co_names):
        """
        Compute the globals used by the function with the given
        bytecode table.
        """
        d = {}
        globs = func.__globals__
        builtins = globs.get('__builtins__', utils.builtins)
        if isinstance(builtins, ModuleType):
            builtins = builtins.__dict__
        for inst in table.values():
            if inst.opname == 'LOAD_GLOBAL':
                name = co_names[_fix_LOAD_GLOBAL_arg(inst.arg)]
                if name not in d:
                    try:
                        value = globs[name]
                    except KeyError:
                        value = builtins[name]
                    d[name] = value
        for co in co_consts:
            if isinstance(co, CodeType):
                subtable = OrderedDict(ByteCodeIter(co))
                d.update(cls._compute_used_globals(func, subtable, co.co_consts, co.co_names))
        return d

    def get_used_globals(self):
        """
        Get a {name: value} map of the globals used by this code
        object and any nested code objects.
        """
        return self._compute_used_globals(self.func_id.func, self.table, self.co_consts, self.co_names)