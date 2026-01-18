from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class ParallelRangeNode(ParallelStatNode):
    """
    This node represents a 'for i in cython.parallel.prange():' construct.

    target       NameNode       the target iteration variable
    else_clause  Node or None   the else clause of this loop
    """
    child_attrs = ['body', 'target', 'else_clause', 'args', 'num_threads', 'chunksize']
    body = target = else_clause = args = None
    start = stop = step = None
    is_prange = True
    nogil = None
    schedule = None
    valid_keyword_arguments = ['schedule', 'nogil', 'num_threads', 'chunksize']

    def __init__(self, pos, **kwds):
        super(ParallelRangeNode, self).__init__(pos, **kwds)
        self.iterator = PassStatNode(pos)

    def analyse_declarations(self, env):
        super(ParallelRangeNode, self).analyse_declarations(env)
        self.target.analyse_target_declaration(env)
        if self.else_clause is not None:
            self.else_clause.analyse_declarations(env)
        if not self.args or len(self.args) > 3:
            error(self.pos, 'Invalid number of positional arguments to prange')
            return
        if len(self.args) == 1:
            self.stop, = self.args
        elif len(self.args) == 2:
            self.start, self.stop = self.args
        else:
            self.start, self.stop, self.step = self.args
        if self.schedule not in (None, 'static', 'dynamic', 'guided', 'runtime'):
            error(self.pos, 'Invalid schedule argument to prange: %s' % (self.schedule,))

    def analyse_expressions(self, env):
        was_nogil = env.nogil
        if self.nogil:
            env.nogil = True
        if self.target is None:
            error(self.pos, 'prange() can only be used as part of a for loop')
            return self
        self.target = self.target.analyse_target_types(env)
        if not self.target.type.is_numeric:
            if not self.target.type.is_pyobject:
                error(self.target.pos, 'Must be of numeric type, not %s' % self.target.type)
            self.index_type = PyrexTypes.c_py_ssize_t_type
        else:
            self.index_type = self.target.type
        self.names = ('start', 'stop', 'step')
        start_stop_step = (self.start, self.stop, self.step)
        for node, name in zip(start_stop_step, self.names):
            if node is not None:
                node.analyse_types(env)
                if not node.type.is_numeric:
                    error(node.pos, '%s argument must be numeric' % name)
                    continue
                if not node.is_literal:
                    node = node.coerce_to_temp(env)
                    setattr(self, name, node)
                self.index_type = PyrexTypes.widest_numeric_type(self.index_type, node.type)
        if self.else_clause is not None:
            self.else_clause = self.else_clause.analyse_expressions(env)
        target_entry = getattr(self.target, 'entry', None)
        if target_entry:
            self.assignments[self.target.entry] = (self.target.pos, None)
        node = super(ParallelRangeNode, self).analyse_expressions(env)
        if node.chunksize:
            if not node.schedule:
                error(node.chunksize.pos, 'Must provide schedule with chunksize')
            elif node.schedule == 'runtime':
                error(node.chunksize.pos, 'Chunksize not valid for the schedule runtime')
            elif node.chunksize.type.is_int and node.chunksize.is_literal and (node.chunksize.compile_time_value(env) <= 0):
                error(node.chunksize.pos, 'Chunksize must not be negative')
            node.chunksize = node.chunksize.coerce_to(PyrexTypes.c_int_type, env).coerce_to_temp(env)
        if node.nogil:
            env.nogil = was_nogil
        node.is_nested_prange = node.parent and node.parent.is_prange
        if node.is_nested_prange:
            parent = node
            while parent.parent and parent.parent.is_prange:
                parent = parent.parent
            parent.assignments.update(node.assignments)
            parent.privates.update(node.privates)
            parent.assigned_nodes.extend(node.assigned_nodes)
        return node

    def nogil_check(self, env):
        names = ('start', 'stop', 'step', 'target')
        nodes = (self.start, self.stop, self.step, self.target)
        for name, node in zip(names, nodes):
            if node is not None and node.type.is_pyobject:
                error(node.pos, "%s may not be a Python object as we don't have the GIL" % name)

    def generate_execution_code(self, code):
        """
        Generate code in the following steps

            1)  copy any closure variables determined thread-private
                into temporaries

            2)  allocate temps for start, stop and step

            3)  generate a loop that calculates the total number of steps,
                which then computes the target iteration variable for every step:

                    for i in prange(start, stop, step):
                        ...

                becomes

                    nsteps = (stop - start) / step;
                    i = start;

                    #pragma omp parallel for lastprivate(i)
                    for (temp = 0; temp < nsteps; temp++) {
                        i = start + step * temp;
                        ...
                    }

                Note that accumulation of 'i' would have a data dependency
                between iterations.

                Also, you can't do this

                    for (i = start; i < stop; i += step)
                        ...

                as the '<' operator should become '>' for descending loops.
                'for i from x < i < y:' does not suffer from this problem
                as the relational operator is known at compile time!

            4) release our temps and write back any private closure variables
        """
        self.declare_closure_privates(code)
        target_index_cname = self.target.entry.cname
        fmt_dict = {'target': target_index_cname, 'target_type': self.target.type.empty_declaration_code()}
        start_stop_step = (self.start, self.stop, self.step)
        defaults = ('0', '0', '1')
        for node, name, default in zip(start_stop_step, self.names, defaults):
            if node is None:
                result = default
            elif node.is_literal:
                result = node.get_constant_c_result_code()
            else:
                node.generate_evaluation_code(code)
                result = node.result()
            fmt_dict[name] = result
        fmt_dict['i'] = code.funcstate.allocate_temp(self.index_type, False)
        fmt_dict['nsteps'] = code.funcstate.allocate_temp(self.index_type, False)
        if self.step is not None and self.step.has_constant_result() and (self.step.constant_result == 0):
            error(node.pos, 'Iteration with step 0 is invalid.')
        elif not fmt_dict['step'].isdigit() or int(fmt_dict['step']) == 0:
            code.putln('if (((%(step)s) == 0)) abort();' % fmt_dict)
        self.setup_parallel_control_flow_block(code)
        code.putln('%(nsteps)s = (%(stop)s - %(start)s + %(step)s - %(step)s/abs(%(step)s)) / %(step)s;' % fmt_dict)
        code.putln('if (%(nsteps)s > 0)' % fmt_dict)
        code.begin_block()
        self.generate_loop(code, fmt_dict)
        code.end_block()
        self.restore_labels(code)
        if self.else_clause:
            if self.breaking_label_used:
                code.put('if (%s < 2)' % Naming.parallel_why)
            code.begin_block()
            code.putln('/* else */')
            self.else_clause.generate_execution_code(code)
            code.end_block()
        self.end_parallel_control_flow_block(code)
        for temp in start_stop_step + (self.chunksize,):
            if temp is not None:
                temp.generate_disposal_code(code)
                temp.free_temps(code)
        code.funcstate.release_temp(fmt_dict['i'])
        code.funcstate.release_temp(fmt_dict['nsteps'])
        self.release_closure_privates(code)

    def generate_loop(self, code, fmt_dict):
        if self.is_nested_prange:
            code.putln('#if 0')
        else:
            code.putln('#ifdef _OPENMP')
        if not self.is_parallel:
            code.put('#pragma omp for')
            self.privatization_insertion_point = code.insertion_point()
            reduction_codepoint = self.parent.privatization_insertion_point
        else:
            code.put('#pragma omp parallel')
            self.privatization_insertion_point = code.insertion_point()
            reduction_codepoint = self.privatization_insertion_point
            code.putln('')
            code.putln('#endif /* _OPENMP */')
            code.begin_block()
            self.begin_parallel_block(code)
            if self.is_nested_prange:
                code.putln('#if 0')
            else:
                code.putln('#ifdef _OPENMP')
            code.put('#pragma omp for')
        for entry, (op, lastprivate) in sorted(self.privates.items()):
            if op and op in '+*-&^|' and (entry != self.target.entry):
                if entry.type.is_pyobject:
                    error(self.pos, 'Python objects cannot be reductions')
                else:
                    reduction_codepoint.put(' reduction(%s:%s)' % (op, entry.cname))
            else:
                if entry == self.target.entry:
                    code.put(' firstprivate(%s)' % entry.cname)
                    code.put(' lastprivate(%s)' % entry.cname)
                    continue
                if not entry.type.is_pyobject:
                    if lastprivate:
                        private = 'lastprivate'
                    else:
                        private = 'private'
                    code.put(' %s(%s)' % (private, entry.cname))
        if self.schedule:
            if self.chunksize:
                chunksize = ', %s' % self.evaluate_before_block(code, self.chunksize)
            else:
                chunksize = ''
            code.put(' schedule(%s%s)' % (self.schedule, chunksize))
        self.put_num_threads(reduction_codepoint)
        code.putln('')
        code.putln('#endif /* _OPENMP */')
        code.put('for (%(i)s = 0; %(i)s < %(nsteps)s; %(i)s++)' % fmt_dict)
        code.begin_block()
        guard_around_body_codepoint = code.insertion_point()
        code.begin_block()
        code.putln('%(target)s = (%(target_type)s)(%(start)s + %(step)s * %(i)s);' % fmt_dict)
        self.initialize_privates_to_nan(code, exclude=self.target.entry)
        if self.is_parallel and (not self.is_nested_prange):
            code.funcstate.start_collecting_temps()
        self.body.generate_execution_code(code)
        self.trap_parallel_exit(code, should_flush=True)
        if self.is_parallel and (not self.is_nested_prange):
            self.privatize_temps(code)
        if self.breaking_label_used:
            guard_around_body_codepoint.putln('if (%s < 2)' % Naming.parallel_why)
        code.end_block()
        code.end_block()
        if self.is_parallel:
            self.end_parallel_block(code)
            code.end_block()