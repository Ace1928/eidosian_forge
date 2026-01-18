import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
@torch.no_grad()
def _check_trace(check_inputs, func, traced_func, check_tolerance, strict, force_outplace, is_trace_module, _module_class, example_inputs_is_kwarg=False):
    for inputs in check_inputs:
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        if is_trace_module:
            copied_dict = {}
            for name, data in inputs.items():
                copied_dict[name] = _clone_inputs(data)
            check_mod = torch.jit.trace_module(getattr(func, '__self__', func), copied_dict, check_trace=False, strict=strict, _force_outplace=force_outplace, _module_class=_module_class, _compilation_unit=torch._C.CompilationUnit(), example_inputs_is_kwarg=example_inputs_is_kwarg, _store_inputs=False)
            check_mod_func = check_mod._c._get_method(traced_func.name)
            inputs = inputs[traced_func.name]
            if isinstance(inputs, torch.Tensor) or (isinstance(inputs, dict) and (not example_inputs_is_kwarg)):
                inputs = (inputs,)
        else:
            if example_inputs_is_kwarg:
                check_mod = torch.jit.trace(func, check_trace=False, strict=strict, _force_outplace=force_outplace, _module_class=_module_class, example_kwarg_inputs=_clone_inputs(inputs), _store_inputs=False)
            else:
                check_mod = torch.jit.trace(func, _clone_inputs(inputs), check_trace=False, strict=strict, _force_outplace=force_outplace, _module_class=_module_class, _store_inputs=False)
            check_mod_func = check_mod

        def graph_diagnostic_info():
            mod_canonicalized = torch._C._jit_pass_canonicalize(traced_func.graph)
            torch._C._jit_pass_inline(mod_canonicalized)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            mod_str = str(mod_canonicalized)
            mod_str = re.sub('___torch_mangle_[0-9]+\\.', '', mod_str)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod_func.graph)
            torch._C._jit_pass_inline(check_canonicalized)
            torch._C._jit_pass_erase_shape_information(check_canonicalized)
            check_str = str(check_canonicalized)
            check_str = re.sub('___torch_mangle_[0-9]+\\.', '', check_str)
            graph_diff_errors = None
            if mod_str != check_str:
                import difflib
                graph_diff = difflib.ndiff(mod_str.splitlines(True), check_str.splitlines(True))
                graph_diff_errors = 'Graph diff:\n' + indent(''.join(graph_diff)) + '\n'
                for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                    if str(n_mod) != str(n_check):
                        graph_diff_errors += 'First diverging operator:\n'
                        node_diff = difflib.ndiff(str(n_mod).splitlines(True), str(n_check).splitlines(True))
                        source_printout = 'Node diff:\n' + indent(''.join(node_diff)) + '\n'
                        mod_stack = n_mod.sourceRange()
                        if mod_stack:
                            source_printout += 'Trace source location:\n' + indent(mod_stack) + '\n'
                        check_stack = n_check.sourceRange()
                        if check_stack:
                            source_printout += 'Check source location:\n' + indent(check_stack) + '\n'
                        graph_diff_errors += source_printout
                        break
            tensor_compare_errors = None
            for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                if n_mod.kind() != n_check.kind():
                    break
                if n_mod.kind() == 'prim::Constant' and (not (n_mod.mustBeNone() or n_check.mustBeNone())):
                    if not n_mod.hasAttribute('value'):
                        continue
                    if n_mod.kindOf('value') != 't' or n_check.kindOf('value') != 't':
                        continue
                    mod_tensor_val = n_mod.t('value')
                    check_tensor_val = n_check.t('value')
                    try:
                        torch.testing.assert_close(mod_tensor_val, check_tensor_val, equal_nan=True)
                    except (RuntimeError, AssertionError) as e:
                        if tensor_compare_errors is None:
                            tensor_compare_errors = ''
                        tensor_compare_errors += 'Node:\n' + indent(str(n_mod)) + '\n'
                        compare_stack = n_mod.sourceRange()
                        if compare_stack:
                            tensor_compare_errors += 'Source Location:\n' + indent(compare_stack) + '\n'
                        tensor_compare_errors += 'Comparison exception: ' + indent(str(e))
                        break
            return (graph_diff_errors, tensor_compare_errors)

        def wrap_retval(x):
            return x if isinstance(x, tuple) else (x,)

        def run_mod_and_filter_tensor_outputs(mod, inputs, running_what):
            try:
                if isinstance(inputs, dict) and example_inputs_is_kwarg:
                    outs = wrap_retval(mod(**inputs))
                else:
                    outs = wrap_retval(mod(*_clone_inputs(inputs)))
                outs = [out for out in outs if isinstance(out, torch.Tensor)]
                return outs
            except Exception as e:
                graph_diff_errors, tensor_compare_errors = graph_diagnostic_info()
                msg = f'encountered an exception while running the {running_what} with test inputs.\nException:\n{indent(str(e))}'
                raise TracingCheckError(graph_diff_errors, tensor_compare_errors, extra_msg=msg) from e
        has_warned = [False]

        def maybe_warn_nondeterministic():
            if has_warned[0]:
                return
            has_warned[0] = True
            nondeterm_ops = [op for op in traced_func.graph.nodes() if op.isNondeterministic()]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = 'Trace had nondeterministic nodes. '
                nondeterministic_ops_warning += 'Did you forget call .eval() on your model? Nodes:\n'
                nondeterministic_ops_warning += '\n'.join([indent(str(op)) for op in nondeterm_ops][:20])
                nondeterministic_ops_warning += '\nThis may cause errors in trace checking. To disable trace checking, pass check_trace=False to torch.jit.trace()'
                warnings.warn(nondeterministic_ops_warning, category=TracerWarning, stacklevel=5)

        def compare_outputs(original, reference, match_what):
            all_ok = True
            for i, (orig, ref) in enumerate(zip(original, reference)):
                try:
                    if orig.is_quantized:
                        orig = orig.dequantize()
                    if ref.is_quantized:
                        ref = ref.dequantize()
                    if orig.is_mkldnn:
                        orig = orig.to_dense()
                    if ref.is_mkldnn:
                        ref = ref.to_dense()
                    if ref.is_complex() or orig.is_complex():
                        torch.testing.assert_close(orig.to(torch.cdouble), ref.to(torch.cdouble), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
                    elif orig.is_mps or ref.is_mps:
                        torch.testing.assert_close(orig.float(), ref.float(), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
                    else:
                        torch.testing.assert_close(orig.double(), ref.double(), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
                except AssertionError as e:
                    maybe_warn_nondeterministic()
                    warnings.warn('Output nr ' + str(i + 1) + '. of the traced function does not match the corresponding output of the ' + match_what + '. Detailed error:\n' + str(e), category=TracerWarning, stacklevel=4)
                    all_ok = False
            return all_ok
        traced_outs = run_mod_and_filter_tensor_outputs(traced_func, inputs, 'trace')
        fn_outs = run_mod_and_filter_tensor_outputs(func, inputs, 'Python function')
        if compare_outputs(traced_outs, fn_outs, 'Python function'):
            check_outs = run_mod_and_filter_tensor_outputs(check_mod_func, inputs, 'repeated trace')
            compare_outputs(traced_outs, check_outs, 'repeated trace')
        diag_info = graph_diagnostic_info()
        if any((info is not None for info in diag_info)):
            raise TracingCheckError(*diag_info)