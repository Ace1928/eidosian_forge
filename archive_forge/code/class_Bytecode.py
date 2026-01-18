import sys
import types
import collections
import io
from opcode import *
from opcode import (
class Bytecode:
    """The bytecode operations of a piece of code

    Instantiate this with a function, method, other compiled object, string of
    code, or a code object (as returned by compile()).

    Iterating over this yields the bytecode operations as Instruction instances.
    """

    def __init__(self, x, *, first_line=None, current_offset=None, show_caches=False, adaptive=False):
        self.codeobj = co = _get_code_object(x)
        if first_line is None:
            self.first_line = co.co_firstlineno
            self._line_offset = 0
        else:
            self.first_line = first_line
            self._line_offset = first_line - co.co_firstlineno
        self._linestarts = dict(findlinestarts(co))
        self._original_object = x
        self.current_offset = current_offset
        self.exception_entries = _parse_exception_table(co)
        self.show_caches = show_caches
        self.adaptive = adaptive

    def __iter__(self):
        co = self.codeobj
        return _get_instructions_bytes(_get_code_array(co, self.adaptive), co._varname_from_oparg, co.co_names, co.co_consts, self._linestarts, line_offset=self._line_offset, exception_entries=self.exception_entries, co_positions=co.co_positions(), show_caches=self.show_caches)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._original_object)

    @classmethod
    def from_traceback(cls, tb, *, show_caches=False, adaptive=False):
        """ Construct a Bytecode from the given traceback """
        while tb.tb_next:
            tb = tb.tb_next
        return cls(tb.tb_frame.f_code, current_offset=tb.tb_lasti, show_caches=show_caches, adaptive=adaptive)

    def info(self):
        """Return formatted information about the code object."""
        return _format_code_info(self.codeobj)

    def dis(self):
        """Return a formatted view of the bytecode operations."""
        co = self.codeobj
        if self.current_offset is not None:
            offset = self.current_offset
        else:
            offset = -1
        with io.StringIO() as output:
            _disassemble_bytes(_get_code_array(co, self.adaptive), varname_from_oparg=co._varname_from_oparg, names=co.co_names, co_consts=co.co_consts, linestarts=self._linestarts, line_offset=self._line_offset, file=output, lasti=offset, exception_entries=self.exception_entries, co_positions=co.co_positions(), show_caches=self.show_caches)
            return output.getvalue()