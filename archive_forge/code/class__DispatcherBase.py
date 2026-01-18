import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
class _DispatcherBase(_dispatcher.Dispatcher):
    """
    Common base class for dispatcher Implementations.
    """
    __numba__ = 'py_func'

    def __init__(self, arg_count, py_func, pysig, can_fallback, exact_match_required):
        self._tm = default_type_manager
        self.overloads = collections.OrderedDict()
        self.py_func = py_func
        self.func_code = get_code_object(py_func)
        self.__code__ = self.func_code
        self._types_active_call = []
        self.__defaults__ = py_func.__defaults__
        argnames = tuple(pysig.parameters)
        default_values = self.py_func.__defaults__ or ()
        defargs = tuple((OmittedArg(val) for val in default_values))
        try:
            lastarg = list(pysig.parameters.values())[-1]
        except IndexError:
            has_stararg = False
        else:
            has_stararg = lastarg.kind == lastarg.VAR_POSITIONAL
        _dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(), arg_count, self._fold_args, argnames, defargs, can_fallback, has_stararg, exact_match_required)
        self.doc = py_func.__doc__
        self._compiling_counter = CompilingCounter()
        weakref.finalize(self, self._make_finalizer())

    def _compilation_chain_init_hook(self):
        """
        This will be called ahead of any part of compilation taking place (this
        even includes being ahead of working out the types of the arguments).
        This permits activities such as initialising extension entry points so
        that the compiler knows about additional externally defined types etc
        before it does anything.
        """
        entrypoints.init_all()

    def _reset_overloads(self):
        self._clear()
        self.overloads.clear()

    def _make_finalizer(self):
        """
        Return a finalizer function that will release references to
        related compiled functions.
        """
        overloads = self.overloads
        targetctx = self.targetctx

        def finalizer(shutting_down=utils.shutting_down):
            if shutting_down():
                return
            for cres in overloads.values():
                try:
                    targetctx.remove_user_function(cres.entry_point)
                except KeyError:
                    pass
        return finalizer

    @property
    def signatures(self):
        """
        Returns a list of compiled function signatures.
        """
        return list(self.overloads)

    @property
    def nopython_signatures(self):
        return [cres.signature for cres in self.overloads.values() if not cres.objectmode]

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time.
        """
        assert not val or len(self.signatures) > 0
        self._can_compile = not val

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        sig = [a._code for a in args]
        self._insert(sig, cres.entry_point, cres.objectmode)
        self.overloads[args] = cres

    def fold_argument_types(self, args, kws):
        return self._compiler.fold_argument_types(args, kws)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows to resolve the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        pysig, args = self._compiler.fold_argument_types(args, kws)
        kws = {}
        if self._can_compile:
            self.compile(tuple(args))
        func_name = self.py_func.__name__
        name = 'CallTemplate({0})'.format(func_name)
        call_template = typing.make_concrete_template(name, key=func_name, signatures=self.nopython_signatures)
        return (call_template, pysig, args, kws)

    def get_overload(self, sig):
        """
        Return the compiled function for the given signature.
        """
        args, return_type = sigutils.normalize_signature(sig)
        return self.overloads[tuple(args)].entry_point

    @property
    def is_compiling(self):
        """
        Whether a specialization is currently being compiled.
        """
        return self._compiling_counter

    def _compile_for_args(self, *args, **kws):
        """
        For internal use.  Compile a specialized version of the function
        for the given *args* and *kws*, and return the resulting callable.
        """
        assert not kws
        self._compilation_chain_init_hook()

        def error_rewrite(e, issue_type):
            """
            Rewrite and raise Exception `e` with help supplied based on the
            specified issue_type.
            """
            if config.SHOW_HELP:
                help_msg = errors.error_extras[issue_type]
                e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
            if config.FULL_TRACEBACKS:
                raise e
            else:
                raise e.with_traceback(None)
        argtypes = []
        for a in args:
            if isinstance(a, OmittedArg):
                argtypes.append(types.Omitted(a.value))
            else:
                argtypes.append(self.typeof_pyval(a))
        return_val = None
        try:
            return_val = self.compile(tuple(argtypes))
        except errors.ForceLiteralArg as e:
            already_lit_pos = [i for i in e.requested_args if isinstance(args[i], types.Literal)]
            if already_lit_pos:
                m = 'Repeated literal typing request.\n{}.\nThis is likely caused by an error in typing. Please see nested and suppressed exceptions.'
                info = ', '.join(('Arg #{} is {}'.format(i, args[i]) for i in sorted(already_lit_pos)))
                raise errors.CompilerError(m.format(info))
            args = [(types.literal if i in e.requested_args else lambda x: x)(args[i]) for i, v in enumerate(args)]
            return_val = self._compile_for_args(*args)
        except errors.TypingError as e:
            failed_args = []
            for i, arg in enumerate(args):
                val = arg.value if isinstance(arg, OmittedArg) else arg
                try:
                    tp = typeof(val, Purpose.argument)
                except ValueError as typeof_exc:
                    failed_args.append((i, str(typeof_exc)))
                else:
                    if tp is None:
                        failed_args.append((i, f'cannot determine Numba type of value {val}'))
            if failed_args:
                args_str = '\n'.join((f'- argument {i}: {err}' for i, err in failed_args))
                msg = f'{str(e).rstrip()} \n\nThis error may have been caused by the following argument(s):\n{args_str}\n'
                e.patch_message(msg)
            error_rewrite(e, 'typing')
        except errors.UnsupportedError as e:
            error_rewrite(e, 'unsupported_error')
        except (errors.NotDefinedError, errors.RedefinedError, errors.VerificationError) as e:
            error_rewrite(e, 'interpreter')
        except errors.ConstantInferenceError as e:
            error_rewrite(e, 'constant_inference')
        except Exception as e:
            if config.SHOW_HELP:
                if hasattr(e, 'patch_message'):
                    help_msg = errors.error_extras['reportable']
                    e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
            raise e
        finally:
            self._types_active_call = []
        return return_val

    def inspect_llvm(self, signature=None):
        """Get the LLVM intermediate representation generated by compilation.

        Parameters
        ----------
        signature : tuple of numba types, optional
            Specify a signature for which to obtain the LLVM IR. If None, the
            IR is returned for all available signatures.

        Returns
        -------
        llvm : dict[signature, str] or str
            Either the LLVM IR string for the specified signature, or, if no
            signature was given, a dictionary mapping signatures to LLVM IR
            strings.
        """
        if signature is not None:
            lib = self.overloads[signature].library
            return lib.get_llvm_str()
        return dict(((sig, self.inspect_llvm(sig)) for sig in self.signatures))

    def inspect_asm(self, signature=None):
        """Get the generated assembly code.

        Parameters
        ----------
        signature : tuple of numba types, optional
            Specify a signature for which to obtain the assembly code. If
            None, the assembly code is returned for all available signatures.

        Returns
        -------
        asm : dict[signature, str] or str
            Either the assembly code for the specified signature, or, if no
            signature was given, a dictionary mapping signatures to assembly
            code.
        """
        if signature is not None:
            lib = self.overloads[signature].library
            return lib.get_asm_str()
        return dict(((sig, self.inspect_asm(sig)) for sig in self.signatures))

    def inspect_types(self, file=None, signature=None, pretty=False, style='default', **kwargs):
        """Print/return Numba intermediate representation (IR)-annotated code.

        Parameters
        ----------
        file : file-like object, optional
            File to which to print. Defaults to sys.stdout if None. Must be
            None if ``pretty=True``.
        signature : tuple of numba types, optional
            Print/return the intermediate representation for only the given
            signature. If None, the IR is printed for all available signatures.
        pretty : bool, optional
            If True, an Annotate object will be returned that can render the
            IR with color highlighting in Jupyter and IPython. ``file`` must
            be None if ``pretty`` is True. Additionally, the ``pygments``
            library must be installed for ``pretty=True``.
        style : str, optional
            Choose a style for rendering. Ignored if ``pretty`` is ``False``.
            This is directly consumed by ``pygments`` formatters. To see a
            list of available styles, import ``pygments`` and run
            ``list(pygments.styles.get_all_styles())``.

        Returns
        -------
        annotated : Annotate object, optional
            Only returned if ``pretty=True``, otherwise this function is only
            used for its printing side effect. If ``pretty=True``, an Annotate
            object is returned that can render itself in Jupyter and IPython.
        """
        overloads = self.overloads
        if signature is not None:
            overloads = {signature: self.overloads[signature]}
        if not pretty:
            if file is None:
                file = sys.stdout
            for ver, res in overloads.items():
                print('%s %s' % (self.py_func.__name__, ver), file=file)
                print('-' * 80, file=file)
                print(res.type_annotation, file=file)
                print('=' * 80, file=file)
        else:
            if file is not None:
                raise ValueError('`file` must be None if `pretty=True`')
            from numba.core.annotations.pretty_annotate import Annotate
            return Annotate(self, signature=signature, style=style)

    def inspect_cfg(self, signature=None, show_wrapper=None, **kwargs):
        """
        For inspecting the CFG of the function.

        By default the CFG of the user function is shown.  The *show_wrapper*
        option can be set to "python" or "cfunc" to show the python wrapper
        function or the *cfunc* wrapper function, respectively.

        Parameters accepted in kwargs
        -----------------------------
        filename : string, optional
            the name of the output file, if given this will write the output to
            filename
        view : bool, optional
            whether to immediately view the optional output file
        highlight : bool, set, dict, optional
            what, if anything, to highlight, options are:
            { incref : bool, # highlight NRT_incref calls
              decref : bool, # highlight NRT_decref calls
              returns : bool, # highlight exits which are normal returns
              raises : bool, # highlight exits which are from raise
              meminfo : bool, # highlight calls to NRT*meminfo
              branches : bool, # highlight true/false branches
             }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {'incref', 'decref'} would
            switch on highlighting on increfs and decrefs.
        interleave: bool, set, dict, optional
            what, if anything, to interleave in the LLVM IR, options are:
            { python: bool # interleave python source code with the LLVM IR
              lineinfo: bool # interleave line information markers with the LLVM
                             # IR
            }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {'python',} would
            switch on interleaving of python source code in the LLVM IR.
        strip_ir : bool, optional
            Default is False. If set to True all LLVM IR that is superfluous to
            that requested in kwarg `highlight` will be removed.
        show_key : bool, optional
            Default is True. Create a "key" for the highlighting in the rendered
            CFG.
        fontsize : int, optional
            Default is 8. Set the fontsize in the output to this value.
        """
        if signature is not None:
            cres = self.overloads[signature]
            lib = cres.library
            if show_wrapper == 'python':
                fname = cres.fndesc.llvm_cpython_wrapper_name
            elif show_wrapper == 'cfunc':
                fname = cres.fndesc.llvm_cfunc_wrapper_name
            else:
                fname = cres.fndesc.mangled_name
            return lib.get_function_cfg(fname, py_func=self.py_func, **kwargs)
        return dict(((sig, self.inspect_cfg(sig, show_wrapper=show_wrapper)) for sig in self.signatures))

    def inspect_disasm_cfg(self, signature=None):
        """
        For inspecting the CFG of the disassembly of the function.

        Requires python package: r2pipe
        Requires radare2 binary on $PATH.
        Notebook rendering requires python package: graphviz

        signature : tuple of Numba types, optional
            Print/return the disassembly CFG for only the given signatures.
            If None, the IR is printed for all available signatures.
        """
        if signature is not None:
            cres = self.overloads[signature]
            lib = cres.library
            return lib.get_disasm_cfg(cres.fndesc.mangled_name)
        return dict(((sig, self.inspect_disasm_cfg(sig)) for sig in self.signatures))

    def get_annotation_info(self, signature=None):
        """
        Gets the annotation information for the function specified by
        signature. If no signature is supplied a dictionary of signature to
        annotation information is returned.
        """
        signatures = self.signatures if signature is None else [signature]
        out = collections.OrderedDict()
        for sig in signatures:
            cres = self.overloads[sig]
            ta = cres.type_annotation
            key = (ta.func_id.filename + ':' + str(ta.func_id.firstlineno + 1), ta.signature)
            out[key] = ta.annotate_raw()[key]
        return out

    def _explain_ambiguous(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        """
        assert not kws, 'kwargs not handled'
        args = tuple([self.typeof_pyval(a) for a in args])
        sigs = self.nopython_signatures
        self.typingctx.resolve_overload(self.py_func, sigs, args, kws, allow_ambiguous=False)

    def _explain_matching_error(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        """
        assert not kws, 'kwargs not handled'
        args = [self.typeof_pyval(a) for a in args]
        msg = 'No matching definition for argument type(s) %s' % ', '.join(map(str, args))
        raise TypeError(msg)

    def _search_new_conversions(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        Search for approximately matching signatures for the given arguments,
        and ensure the corresponding conversions are registered in the C++
        type manager.
        """
        assert not kws, 'kwargs not handled'
        args = [self.typeof_pyval(a) for a in args]
        found = False
        for sig in self.nopython_signatures:
            conv = self.typingctx.install_possible_conversions(args, sig.args)
            if conv:
                found = True
        return found

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self.py_func)

    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
        try:
            tp = typeof(val, Purpose.argument)
        except ValueError:
            tp = types.pyobject
        else:
            if tp is None:
                tp = types.pyobject
        self._types_active_call.append(tp)
        return tp

    def _callback_add_timer(self, duration, cres, lock_name):
        md = cres.metadata
        if md is not None:
            timers = md.setdefault('timers', {})
            if lock_name not in timers:
                timers[lock_name] = duration
            else:
                msg = f"'{lock_name} metadata is already defined."
                raise AssertionError(msg)

    def _callback_add_compiler_timer(self, duration, cres):
        return self._callback_add_timer(duration, cres, lock_name='compiler_lock')

    def _callback_add_llvm_timer(self, duration, cres):
        return self._callback_add_timer(duration, cres, lock_name='llvm_lock')