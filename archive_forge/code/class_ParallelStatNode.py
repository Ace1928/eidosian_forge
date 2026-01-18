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
class ParallelStatNode(StatNode, ParallelNode):
    """
    Base class for 'with cython.parallel.parallel():' and 'for i in prange():'.

    assignments     { Entry(var) : (var.pos, inplace_operator_or_None) }
                    assignments to variables in this parallel section

    parent          parent ParallelStatNode or None
    is_parallel     indicates whether this node is OpenMP parallel
                    (true for #pragma omp parallel for and
                              #pragma omp parallel)

    is_parallel is true for:

        #pragma omp parallel
        #pragma omp parallel for

    sections, but NOT for

        #pragma omp for

    We need this to determine the sharing attributes.

    privatization_insertion_point   a code insertion point used to make temps
                                    private (esp. the "nsteps" temp)

    args         tuple          the arguments passed to the parallel construct
    kwargs       DictNode       the keyword arguments passed to the parallel
                                construct (replaced by its compile time value)
    """
    child_attrs = ['body', 'num_threads']
    body = None
    is_prange = False
    is_nested_prange = False
    error_label_used = False
    num_threads = None
    chunksize = None
    parallel_exc = (Naming.parallel_exc_type, Naming.parallel_exc_value, Naming.parallel_exc_tb)
    parallel_pos_info = (Naming.parallel_filename, Naming.parallel_lineno, Naming.parallel_clineno)
    pos_info = (Naming.filename_cname, Naming.lineno_cname, Naming.clineno_cname)
    critical_section_counter = 0

    def __init__(self, pos, **kwargs):
        super(ParallelStatNode, self).__init__(pos, **kwargs)
        self.assignments = kwargs.get('assignments') or {}
        self.seen_closure_vars = set()
        self.privates = {}
        self.assigned_nodes = []

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)
        self.num_threads = None
        if self.kwargs:
            pairs = []
            seen = set()
            for dictitem in self.kwargs.key_value_pairs:
                if dictitem.key.value in seen:
                    error(self.pos, 'Duplicate keyword argument found: %s' % dictitem.key.value)
                seen.add(dictitem.key.value)
                if dictitem.key.value == 'num_threads':
                    if not dictitem.value.is_none:
                        self.num_threads = dictitem.value
                elif self.is_prange and dictitem.key.value == 'chunksize':
                    if not dictitem.value.is_none:
                        self.chunksize = dictitem.value
                else:
                    pairs.append(dictitem)
            self.kwargs.key_value_pairs = pairs
            try:
                self.kwargs = self.kwargs.compile_time_value(env)
            except Exception as e:
                error(self.kwargs.pos, 'Only compile-time values may be supplied as keyword arguments')
        else:
            self.kwargs = {}
        for kw, val in self.kwargs.items():
            if kw not in self.valid_keyword_arguments:
                error(self.pos, 'Invalid keyword argument: %s' % kw)
            else:
                setattr(self, kw, val)

    def analyse_expressions(self, env):
        if self.num_threads:
            self.num_threads = self.num_threads.analyse_expressions(env)
        if self.chunksize:
            self.chunksize = self.chunksize.analyse_expressions(env)
        self.body = self.body.analyse_expressions(env)
        self.analyse_sharing_attributes(env)
        if self.num_threads is not None:
            if self.parent and self.parent.num_threads is not None and (not self.parent.is_prange):
                error(self.pos, 'num_threads already declared in outer section')
            elif self.parent and (not self.parent.is_prange):
                error(self.pos, 'num_threads must be declared in the parent parallel section')
            elif self.num_threads.type.is_int and self.num_threads.is_literal and (self.num_threads.compile_time_value(env) <= 0):
                error(self.pos, 'argument to num_threads must be greater than 0')
            if not self.num_threads.is_simple() or self.num_threads.type.is_pyobject:
                self.num_threads = self.num_threads.coerce_to(PyrexTypes.c_int_type, env).coerce_to_temp(env)
        return self

    def analyse_sharing_attributes(self, env):
        """
        Analyse the privates for this block and set them in self.privates.
        This should be called in a post-order fashion during the
        analyse_expressions phase
        """
        for entry, (pos, op) in self.assignments.items():
            if self.is_prange and (not self.is_parallel):
                if entry in self.parent.assignments:
                    error(pos, 'Cannot assign to private of outer parallel block')
                    continue
            if not self.is_prange and op:
                error(pos, 'Reductions not allowed for parallel blocks')
                continue
            lastprivate = True
            self.propagate_var_privatization(entry, pos, op, lastprivate)

    def propagate_var_privatization(self, entry, pos, op, lastprivate):
        """
        Propagate the sharing attributes of a variable. If the privatization is
        determined by a parent scope, done propagate further.

        If we are a prange, we propagate our sharing attributes outwards to
        other pranges. If we are a prange in parallel block and the parallel
        block does not determine the variable private, we propagate to the
        parent of the parent. Recursion stops at parallel blocks, as they have
        no concept of lastprivate or reduction.

        So the following cases propagate:

            sum is a reduction for all loops:

                for i in prange(n):
                    for j in prange(n):
                        for k in prange(n):
                            sum += i * j * k

            sum is a reduction for both loops, local_var is private to the
            parallel with block:

                for i in prange(n):
                    with parallel:
                        local_var = ... # private to the parallel
                        for j in prange(n):
                            sum += i * j

        Nested with parallel blocks are disallowed, because they wouldn't
        allow you to propagate lastprivates or reductions:

            #pragma omp parallel for lastprivate(i)
            for i in prange(n):

                sum = 0

                #pragma omp parallel private(j, sum)
                with parallel:

                    #pragma omp parallel
                    with parallel:

                        #pragma omp for lastprivate(j) reduction(+:sum)
                        for j in prange(n):
                            sum += i

                    # sum and j are well-defined here

                # sum and j are undefined here

            # sum and j are undefined here
        """
        self.privates[entry] = (op, lastprivate)
        if entry.type.is_memoryviewslice:
            error(pos, 'Memoryview slices can only be shared in parallel sections')
            return
        if self.is_prange:
            if not self.is_parallel and entry not in self.parent.assignments:
                parent = self.parent.parent
            else:
                parent = self.parent
            if parent and (op or lastprivate):
                parent.propagate_var_privatization(entry, pos, op, lastprivate)

    def _allocate_closure_temp(self, code, entry):
        """
        Helper function that allocate a temporary for a closure variable that
        is assigned to.
        """
        if self.parent:
            return self.parent._allocate_closure_temp(code, entry)
        if entry.cname in self.seen_closure_vars:
            return entry.cname
        cname = code.funcstate.allocate_temp(entry.type, True)
        self.seen_closure_vars.add(entry.cname)
        self.seen_closure_vars.add(cname)
        self.modified_entries.append((entry, entry.cname))
        code.putln('%s = %s;' % (cname, entry.cname))
        entry.cname = cname

    def initialize_privates_to_nan(self, code, exclude=None):
        first = True
        for entry, (op, lastprivate) in sorted(self.privates.items()):
            if not op and (not exclude or entry != exclude):
                invalid_value = entry.type.invalid_value()
                if invalid_value:
                    if first:
                        code.putln('/* Initialize private variables to invalid values */')
                        first = False
                    code.putln('%s = %s;' % (entry.cname, entry.type.cast_code(invalid_value)))

    def evaluate_before_block(self, code, expr):
        c = self.begin_of_parallel_control_block_point_after_decls
        owner = c.funcstate.owner
        c.funcstate.owner = c
        expr.generate_evaluation_code(c)
        c.funcstate.owner = owner
        return expr.result()

    def put_num_threads(self, code):
        """
        Write self.num_threads if set as the num_threads OpenMP directive
        """
        if self.num_threads is not None:
            code.put(' num_threads(%s)' % self.evaluate_before_block(code, self.num_threads))

    def declare_closure_privates(self, code):
        """
        If a variable is in a scope object, we need to allocate a temp and
        assign the value from the temp to the variable in the scope object
        after the parallel section. This kind of copying should be done only
        in the outermost parallel section.
        """
        self.modified_entries = []
        for entry in sorted(self.assignments):
            if entry.from_closure or entry.in_closure:
                self._allocate_closure_temp(code, entry)

    def release_closure_privates(self, code):
        """
        Release any temps used for variables in scope objects. As this is the
        outermost parallel block, we don't need to delete the cnames from
        self.seen_closure_vars.
        """
        for entry, original_cname in self.modified_entries:
            code.putln('%s = %s;' % (original_cname, entry.cname))
            code.funcstate.release_temp(entry.cname)
            entry.cname = original_cname

    def privatize_temps(self, code, exclude_temps=()):
        """
        Make any used temporaries private. Before the relevant code block
        code.start_collecting_temps() should have been called.
        """
        c = self.privatization_insertion_point
        self.privatization_insertion_point = None
        if self.is_parallel:
            self.temps = temps = code.funcstate.stop_collecting_temps()
            privates, firstprivates = ([], [])
            for temp, type in sorted(temps):
                if type.is_pyobject or type.is_memoryviewslice:
                    firstprivates.append(temp)
                else:
                    privates.append(temp)
            if privates:
                c.put(' private(%s)' % ', '.join(privates))
            if firstprivates:
                c.put(' firstprivate(%s)' % ', '.join(firstprivates))
            if self.breaking_label_used:
                shared_vars = [Naming.parallel_why]
                if self.error_label_used:
                    shared_vars.extend(self.parallel_exc)
                    c.put(' private(%s, %s, %s)' % self.pos_info)
                c.put(' shared(%s)' % ', '.join(shared_vars))

    def cleanup_temps(self, code):
        if self.is_parallel and (not self.is_nested_prange):
            code.putln('/* Clean up any temporaries */')
            for temp, type in sorted(self.temps):
                code.put_xdecref_clear(temp, type, have_gil=False)

    def setup_parallel_control_flow_block(self, code):
        """
        Sets up a block that surrounds the parallel block to determine
        how the parallel section was exited. Any kind of return is
        trapped (break, continue, return, exceptions). This is the idea:

        {
            int why = 0;

            #pragma omp parallel
            {
                return # -> goto new_return_label;
                goto end_parallel;

            new_return_label:
                why = 3;
                goto end_parallel;

            end_parallel:;
                #pragma omp flush(why) # we need to flush for every iteration
            }

            if (why == 3)
                goto old_return_label;
        }
        """
        self.old_loop_labels = code.new_loop_labels()
        self.old_error_label = code.new_error_label()
        self.old_return_label = code.return_label
        code.return_label = code.new_label(name='return')
        code.begin_block()
        self.begin_of_parallel_control_block_point = code.insertion_point()
        self.begin_of_parallel_control_block_point_after_decls = code.insertion_point()
        self.undef_builtin_expect_apple_gcc_bug(code)

    def begin_parallel_block(self, code):
        """
        Each OpenMP thread in a parallel section that contains a with gil block
        must have the thread-state initialized. The call to
        PyGILState_Release() then deallocates our threadstate. If we wouldn't
        do this, each with gil block would allocate and deallocate one, thereby
        losing exception information before it can be saved before leaving the
        parallel section.
        """
        self.begin_of_parallel_block = code.insertion_point()

    def end_parallel_block(self, code):
        """
        To ensure all OpenMP threads have thread states, we ensure the GIL
        in each thread (which creates a thread state if it doesn't exist),
        after which we release the GIL.
        On exit, reacquire the GIL and release the thread state.

        If compiled without OpenMP support (at the C level), then we still have
        to acquire the GIL to decref any object temporaries.
        """
        begin_code = self.begin_of_parallel_block
        self.begin_of_parallel_block = None
        if self.error_label_used:
            end_code = code
            begin_code.putln('#ifdef _OPENMP')
            begin_code.put_ensure_gil(declare_gilstate=True)
            begin_code.putln('Py_BEGIN_ALLOW_THREADS')
            begin_code.putln('#endif /* _OPENMP */')
            end_code.putln('#ifdef _OPENMP')
            end_code.putln('Py_END_ALLOW_THREADS')
            end_code.putln('#else')
            end_code.put_safe('{\n')
            end_code.put_ensure_gil()
            end_code.putln('#endif /* _OPENMP */')
            self.cleanup_temps(end_code)
            end_code.put_release_ensured_gil()
            end_code.putln('#ifndef _OPENMP')
            end_code.put_safe('}\n')
            end_code.putln('#endif /* _OPENMP */')

    def trap_parallel_exit(self, code, should_flush=False):
        """
        Trap any kind of return inside a parallel construct. 'should_flush'
        indicates whether the variable should be flushed, which is needed by
        prange to skip the loop. It also indicates whether we need to register
        a continue (we need this for parallel blocks, but not for prange
        loops, as it is a direct jump there).

        It uses the same mechanism as try/finally:
            1 continue
            2 break
            3 return
            4 error
        """
        save_lastprivates_label = code.new_label()
        dont_return_label = code.new_label()
        self.any_label_used = False
        self.breaking_label_used = False
        self.error_label_used = False
        self.parallel_private_temps = []
        all_labels = code.get_all_labels()
        for label in all_labels:
            if code.label_used(label):
                self.breaking_label_used = self.breaking_label_used or label != code.continue_label
                self.any_label_used = True
        if self.any_label_used:
            code.put_goto(dont_return_label)
        for i, label in enumerate(all_labels):
            if not code.label_used(label):
                continue
            is_continue_label = label == code.continue_label
            code.put_label(label)
            if not (should_flush and is_continue_label):
                if label == code.error_label:
                    self.error_label_used = True
                    self.fetch_parallel_exception(code)
                code.putln('%s = %d;' % (Naming.parallel_why, i + 1))
            if self.breaking_label_used and self.is_prange and (not is_continue_label):
                code.put_goto(save_lastprivates_label)
            else:
                code.put_goto(dont_return_label)
        if self.any_label_used:
            if self.is_prange and self.breaking_label_used:
                code.put_label(save_lastprivates_label)
                self.save_parallel_vars(code)
            code.put_label(dont_return_label)
            if should_flush and self.breaking_label_used:
                code.putln_openmp('#pragma omp flush(%s)' % Naming.parallel_why)

    def save_parallel_vars(self, code):
        """
        The following shenanigans are instated when we break, return or
        propagate errors from a prange. In this case we cannot rely on
        lastprivate() to do its job, as no iterations may have executed yet
        in the last thread, leaving the values undefined. It is most likely
        that the breaking thread has well-defined values of the lastprivate
        variables, so we keep those values.
        """
        section_name = '__pyx_parallel_lastprivates%d' % self.critical_section_counter
        code.putln_openmp('#pragma omp critical(%s)' % section_name)
        ParallelStatNode.critical_section_counter += 1
        code.begin_block()
        c = self.begin_of_parallel_control_block_point
        temp_count = 0
        for entry, (op, lastprivate) in sorted(self.privates.items()):
            if not lastprivate or entry.type.is_pyobject:
                continue
            if entry.type.is_cpp_class and (not entry.type.is_fake_reference) and code.globalstate.directives['cpp_locals']:
                type_decl = entry.type.cpp_optional_declaration_code('')
            else:
                type_decl = entry.type.empty_declaration_code()
            temp_cname = '__pyx_parallel_temp%d' % temp_count
            private_cname = entry.cname
            temp_count += 1
            invalid_value = entry.type.invalid_value()
            if invalid_value:
                init = ' = ' + entry.type.cast_code(invalid_value)
            else:
                init = ''
            c.putln('%s %s%s;' % (type_decl, temp_cname, init))
            self.parallel_private_temps.append((temp_cname, private_cname, entry.type))
            if entry.type.is_cpp_class:
                code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
                private_cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % private_cname
            code.putln('%s = %s;' % (temp_cname, private_cname))
        code.end_block()

    def fetch_parallel_exception(self, code):
        """
        As each OpenMP thread may raise an exception, we need to fetch that
        exception from the threadstate and save it for after the parallel
        section where it can be re-raised in the master thread.

        Although it would seem that __pyx_filename, __pyx_lineno and
        __pyx_clineno are only assigned to under exception conditions (i.e.,
        when we have the GIL), and thus should be allowed to be shared without
        any race condition, they are in fact subject to the same race
        conditions that they were previously when they were global variables
        and functions were allowed to release the GIL:

            thread A                thread B
                acquire
                set lineno
                release
                                        acquire
                                        set lineno
                                        release
                acquire
                fetch exception
                release
                                        skip the fetch

                deallocate threadstate  deallocate threadstate
        """
        code.begin_block()
        code.put_ensure_gil(declare_gilstate=True)
        code.putln_openmp('#pragma omp flush(%s)' % Naming.parallel_exc_type)
        code.putln('if (!%s) {' % Naming.parallel_exc_type)
        code.putln('__Pyx_ErrFetchWithState(&%s, &%s, &%s);' % self.parallel_exc)
        pos_info = chain(*zip(self.parallel_pos_info, self.pos_info))
        code.funcstate.uses_error_indicator = True
        code.putln('%s = %s; %s = %s; %s = %s;' % tuple(pos_info))
        code.put_gotref(Naming.parallel_exc_type, py_object_type)
        code.putln('}')
        code.put_release_ensured_gil()
        code.end_block()

    def restore_parallel_exception(self, code):
        """Re-raise a parallel exception"""
        code.begin_block()
        code.put_ensure_gil(declare_gilstate=True)
        code.put_giveref(Naming.parallel_exc_type, py_object_type)
        code.putln('__Pyx_ErrRestoreWithState(%s, %s, %s);' % self.parallel_exc)
        pos_info = chain(*zip(self.pos_info, self.parallel_pos_info))
        code.putln('%s = %s; %s = %s; %s = %s;' % tuple(pos_info))
        code.put_release_ensured_gil()
        code.end_block()

    def restore_labels(self, code):
        """
        Restore all old labels. Call this before the 'else' clause to for
        loops and always before ending the parallel control flow block.
        """
        code.set_all_labels(self.old_loop_labels + (self.old_return_label, self.old_error_label))

    def end_parallel_control_flow_block(self, code, break_=False, continue_=False, return_=False):
        """
        This ends the parallel control flow block and based on how the parallel
        section was exited, takes the corresponding action. The break_ and
        continue_ parameters indicate whether these should be propagated
        outwards:

            for i in prange(...):
                with cython.parallel.parallel():
                    continue

        Here break should be trapped in the parallel block, and propagated to
        the for loop.
        """
        c = self.begin_of_parallel_control_block_point
        self.begin_of_parallel_control_block_point = None
        self.begin_of_parallel_control_block_point_after_decls = None
        if self.num_threads is not None:
            self.num_threads.generate_disposal_code(code)
            self.num_threads.free_temps(code)
        if self.error_label_used:
            c.putln('const char *%s = NULL; int %s = 0, %s = 0;' % self.parallel_pos_info)
            c.putln('PyObject *%s = NULL, *%s = NULL, *%s = NULL;' % self.parallel_exc)
            code.putln('if (%s) {' % Naming.parallel_exc_type)
            code.putln('/* This may have been overridden by a continue, break or return in another thread. Prefer the error. */')
            code.putln('%s = 4;' % Naming.parallel_why)
            code.putln('}')
        if continue_:
            any_label_used = self.any_label_used
        else:
            any_label_used = self.breaking_label_used
        if any_label_used:
            c.putln('int %s;' % Naming.parallel_why)
            c.putln('%s = 0;' % Naming.parallel_why)
            code.putln('if (%s) {' % Naming.parallel_why)
            for temp_cname, private_cname, temp_type in self.parallel_private_temps:
                if temp_type.is_cpp_class:
                    temp_cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % temp_cname
                code.putln('%s = %s;' % (private_cname, temp_cname))
            code.putln('switch (%s) {' % Naming.parallel_why)
            if continue_:
                code.put('    case 1: ')
                code.put_goto(code.continue_label)
            if break_:
                code.put('    case 2: ')
                code.put_goto(code.break_label)
            if return_:
                code.put('    case 3: ')
                code.put_goto(code.return_label)
            if self.error_label_used:
                code.globalstate.use_utility_code(restore_exception_utility_code)
                code.putln('    case 4:')
                self.restore_parallel_exception(code)
                code.put_goto(code.error_label)
            code.putln('}')
            code.putln('}')
        code.end_block()
        self.redef_builtin_expect_apple_gcc_bug(code)
    buggy_platform_macro_condition = '(defined(__APPLE__) || defined(__OSX__))'
    have_expect_condition = '(defined(__GNUC__) && (__GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95))))'
    redef_condition = '(%s && %s)' % (buggy_platform_macro_condition, have_expect_condition)

    def undef_builtin_expect_apple_gcc_bug(self, code):
        """
        A bug on OS X Lion disallows __builtin_expect macros. This code avoids them
        """
        if not self.parent:
            code.undef_builtin_expect(self.redef_condition)

    def redef_builtin_expect_apple_gcc_bug(self, code):
        if not self.parent:
            code.redef_builtin_expect(self.redef_condition)