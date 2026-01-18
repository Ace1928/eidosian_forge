from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _model_to_graph(model, args, verbose=False, input_names=None, output_names=None, operator_export_type=_C_onnx.OperatorExportTypes.ONNX, do_constant_folding=True, _disable_torch_constant_prop=False, fixed_batch_size=False, training=_C_onnx.TrainingMode.EVAL, dynamic_axes=None) -> Tuple[_C.Graph, Dict[str, torch.Tensor], Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor], Any]]]:
    """Converts model into an ONNX graph.

    Returns:
        graph: A TorchScript IR Graph with ONNX nodes.
        params_dict: Dict from input param name to param value.
        torch_out: The output tensors resulting from the trace of ``model``.
            If ``model`` is a :class:`torch.jit.ScriptModule` or :class:`torch.jit.ScriptFunction`,
            this will be None, since we are not doing any tracing.
    """
    if isinstance(args, (torch.Tensor, int, float, bool)):
        args = (args,)
    model = _pre_trace_quant_model(model, args)
    graph, params, torch_out, module = _create_jit_graph(model, args)
    params_dict = _get_named_param_dict(graph, params)
    try:
        graph = _optimize_graph(graph, operator_export_type, _disable_torch_constant_prop=_disable_torch_constant_prop, fixed_batch_size=fixed_batch_size, params_dict=params_dict, dynamic_axes=dynamic_axes, input_names=input_names, module=module)
    except Exception as e:
        torch.onnx.log('Torch IR graph at exception: ', graph)
        raise
    is_script = isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule))
    if is_script:
        example_outputs = _get_example_outputs(model, args)
        example_outputs_final = ()
        for example_output in example_outputs:
            example_outputs_final += unpack_quantized_tensor(example_output)
        out_vars, desc = torch.jit._flatten(example_outputs_final)
        _C._jit_pass_onnx_assign_output_shape(graph, out_vars, desc, GLOBALS.onnx_shape_inference, is_script, GLOBALS.export_onnx_opset_version)
    else:
        if not isinstance(torch_out, (list, tuple)):
            output_wrapped = [torch_out]
        else:
            output_wrapped = torch_out
        output_tensors, out_desc = torch.jit._flatten(tuple(output_wrapped))
        if not any((getattr(out, 'is_quantized', False) for out in output_tensors)):
            _C._jit_pass_onnx_assign_output_shape(graph, output_tensors, out_desc, GLOBALS.onnx_shape_inference, is_script, GLOBALS.export_onnx_opset_version)
    _set_input_and_output_names(graph, input_names, output_names)
    params_dict = _get_named_param_dict(graph, params)
    if do_constant_folding and GLOBALS.export_onnx_opset_version >= _constants.ONNX_CONSTANT_FOLDING_MIN_OPSET:
        if training is None or training == _C_onnx.TrainingMode.EVAL:
            params_dict = _C._jit_pass_onnx_eval_peephole(graph, params_dict)
        params_dict = _C._jit_pass_onnx_constant_fold(graph, params_dict, GLOBALS.export_onnx_opset_version)
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    if GLOBALS.onnx_shape_inference:
        try:
            _C._jit_pass_onnx_graph_shape_type_inference(graph, params_dict, GLOBALS.export_onnx_opset_version)
        except RuntimeError as exc:
            if _C_onnx._CAFFE2_ATEN_FALLBACK and exc.args[0] == 'ScalarType UNKNOWN_SCALAR is an unexpected tensor scalar type!':
                pass
    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)
    if GLOBALS.export_onnx_opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)
    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)
    _C._jit_decay_packed_param_input_types(graph)
    _apply_friendly_debug_names(graph, params_dict)
    return (graph, params_dict, torch_out)