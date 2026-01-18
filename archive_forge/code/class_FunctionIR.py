from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
class FunctionIR(object):

    def __init__(self, blocks, is_generator, func_id, loc, definitions, arg_count, arg_names):
        self.blocks = blocks
        self.is_generator = is_generator
        self.func_id = func_id
        self.loc = loc
        self.arg_count = arg_count
        self.arg_names = arg_names
        self._definitions = definitions
        self._reset_analysis_variables()

    def equal_ir(self, other):
        """ Checks that the IR contained within is equal to the IR in other.
        Equality is defined by being equal in fundamental structure (blocks,
        labels, IR node type and the order in which they are defined) and the
        IR nodes being equal. IR node equality essentially comes down to
        ensuring a node's `.__dict__` or `.__slots__` is equal, with the
        exception of ignoring 'loc' and 'scope' entries. The upshot is that the
        comparison is essentially location and scope invariant, but otherwise
        behaves as unsurprisingly as possible.
        """
        if type(self) is type(other):
            return self.blocks == other.blocks
        return False

    def diff_str(self, other):
        """
        Compute a human readable difference in the IR, returns a formatted
        string ready for printing.
        """
        msg = []
        for label, block in self.blocks.items():
            other_blk = other.blocks.get(label, None)
            if other_blk is not None:
                if block != other_blk:
                    msg.append(('Block %s differs' % label).center(80, '-'))
                    block_del = [x for x in block.body if isinstance(x, Del)]
                    oth_del = [x for x in other_blk.body if isinstance(x, Del)]
                    if block_del != oth_del:
                        if sorted(block_del) == sorted(oth_del):
                            msg.append('Block %s contains the same dels but their order is different' % label)
                    if len(block.body) > len(other_blk.body):
                        msg.append('This block contains more statements')
                    elif len(block.body) < len(other_blk.body):
                        msg.append('Other block contains more statements')
                    tmp = []
                    for idx, stmts in enumerate(zip(block.body, other_blk.body)):
                        b_s, o_s = stmts
                        if b_s != o_s:
                            tmp.append(idx)

                    def get_pad(ablock, l):
                        pointer = '-> '
                        sp = len(pointer) * ' '
                        pad = []
                        nstmt = len(ablock)
                        for i in range(nstmt):
                            if i in tmp:
                                item = pointer
                            elif i >= l:
                                item = pointer
                            else:
                                item = sp
                            pad.append(item)
                        return pad
                    min_stmt_len = min(len(block.body), len(other_blk.body))
                    with StringIO() as buf:
                        it = [('self', block), ('other', other_blk)]
                        for name, _block in it:
                            buf.truncate(0)
                            _block.dump(file=buf)
                            stmts = buf.getvalue().splitlines()
                            pad = get_pad(_block.body, min_stmt_len)
                            title = '%s: block %s' % (name, label)
                            msg.append(title.center(80, '-'))
                            msg.extend(['{0}{1}'.format(a, b) for a, b in zip(pad, stmts)])
        if msg == []:
            msg.append('IR is considered equivalent.')
        return '\n'.join(msg)

    def _reset_analysis_variables(self):
        self._consts = consts.ConstantInference(self)
        self.generator_info = None
        self.variable_lifetime = None
        self.block_entry_vars = {}

    def derive(self, blocks, arg_count=None, arg_names=None, force_non_generator=False):
        """
        Derive a new function IR from this one, using the given blocks,
        and possibly modifying the argument count and generator flag.

        Post-processing will have to be run again on the new IR.
        """
        firstblock = blocks[min(blocks)]
        new_ir = copy.copy(self)
        new_ir.blocks = blocks
        new_ir.loc = firstblock.loc
        if force_non_generator:
            new_ir.is_generator = False
        if arg_count is not None:
            new_ir.arg_count = arg_count
        if arg_names is not None:
            new_ir.arg_names = arg_names
        new_ir._reset_analysis_variables()
        new_ir.func_id = new_ir.func_id.derive()
        return new_ir

    def copy(self):
        new_ir = copy.copy(self)
        blocks = {}
        block_entry_vars = {}
        for label, block in self.blocks.items():
            new_block = block.copy()
            blocks[label] = new_block
            if block in self.block_entry_vars:
                block_entry_vars[new_block] = self.block_entry_vars[block]
        new_ir.blocks = blocks
        new_ir.block_entry_vars = block_entry_vars
        return new_ir

    def get_block_entry_vars(self, block):
        """
        Return a set of variable names possibly alive at the beginning of
        the block.
        """
        return self.block_entry_vars[block]

    def infer_constant(self, name):
        """
        Try to infer the constant value of a given variable.
        """
        if isinstance(name, Var):
            name = name.name
        return self._consts.infer_constant(name)

    def get_definition(self, value, lhs_only=False):
        """
        Get the definition site for the given variable name or instance.
        A Expr instance is returned by default, but if lhs_only is set
        to True, the left-hand-side variable is returned instead.
        """
        lhs = value
        while True:
            if isinstance(value, Var):
                lhs = value
                name = value.name
            elif isinstance(value, str):
                lhs = value
                name = value
            else:
                return lhs if lhs_only else value
            defs = self._definitions[name]
            if len(defs) == 0:
                raise KeyError('no definition for %r' % (name,))
            if len(defs) > 1:
                raise KeyError('more than one definition for %r' % (name,))
            value = defs[0]

    def get_assignee(self, rhs_value, in_blocks=None):
        """
        Finds the assignee for a given RHS value. If in_blocks is given the
        search will be limited to the specified blocks.
        """
        if in_blocks is None:
            blocks = self.blocks.values()
        elif isinstance(in_blocks, int):
            blocks = [self.blocks[in_blocks]]
        else:
            blocks = [self.blocks[blk] for blk in list(in_blocks)]
        assert isinstance(rhs_value, AbstractRHS)
        for blk in blocks:
            for assign in blk.find_insts(Assign):
                if assign.value == rhs_value:
                    return assign.target
        raise ValueError('Could not find an assignee for %s' % rhs_value)

    def dump(self, file=None):
        nofile = file is None
        file = file or StringIO()
        for offset, block in sorted(self.blocks.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file=file)
        if nofile:
            text = file.getvalue()
            if config.HIGHLIGHT_DUMPS:
                try:
                    import pygments
                except ImportError:
                    msg = 'Please install pygments to see highlighted dumps'
                    raise ValueError(msg)
                else:
                    from pygments import highlight
                    from numba.misc.dump_style import NumbaIRLexer as lexer
                    from numba.misc.dump_style import by_colorscheme
                    from pygments.formatters import Terminal256Formatter
                    print(highlight(text, lexer(), Terminal256Formatter(style=by_colorscheme())))
            else:
                print(text)

    def dump_to_string(self):
        with StringIO() as sb:
            self.dump(file=sb)
            return sb.getvalue()

    def dump_generator_info(self, file=None):
        file = file or sys.stdout
        gi = self.generator_info
        print('generator state variables:', sorted(gi.state_vars), file=file)
        for index, yp in sorted(gi.yield_points.items()):
            print('yield point #%d: live variables = %s, weak live variables = %s' % (index, sorted(yp.live_vars), sorted(yp.weak_live_vars)), file=file)

    def render_dot(self, filename_prefix='numba_ir', include_ir=True):
        """Render the CFG of the IR with GraphViz DOT via the
        ``graphviz`` python binding.

        Returns
        -------
        g : graphviz.Digraph
            Use `g.view()` to open the graph in the default PDF application.
        """
        try:
            import graphviz as gv
        except ImportError:
            raise ImportError('The feature requires `graphviz` but it is not available. Please install with `pip install graphviz`')
        g = gv.Digraph(filename='{}{}.dot'.format(filename_prefix, self.func_id.unique_name))
        for k, blk in self.blocks.items():
            with StringIO() as sb:
                blk.dump(sb)
                label = sb.getvalue()
            if include_ir:
                label = ''.join(['  {}\\l'.format(x) for x in label.splitlines()])
                label = 'block {}\\l'.format(k) + label
                g.node(str(k), label=label, shape='rect')
            else:
                label = '{}\\l'.format(k)
                g.node(str(k), label=label, shape='circle')
        for src, blk in self.blocks.items():
            for dst in blk.terminator.get_targets():
                g.edge(str(src), str(dst))
        return g