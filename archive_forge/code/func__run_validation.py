import copy
import gc
import multiprocessing as mp
import os
import traceback
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.modeling_utils import get_parameter_dtype
from transformers.utils import is_tf_available, is_torch_available
from ...onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from ...utils import (
from ...utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, MinimumVersionError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import OnnxConfig
from .constants import UNPICKABLE_ARCHS
from .model_configs import SpeechT5OnnxConfig
from .utils import (
def _run_validation(config: OnnxConfig, reference_model: Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], onnx_model: Path, onnx_named_outputs: List[str], atol: Optional[float]=None, input_shapes: Optional[Dict]=None, device: str='cpu', model_kwargs: Optional[Dict[str, Any]]=None):
    from onnxruntime import GraphOptimizationLevel, SessionOptions
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    logger.info(f'Validating ONNX model {onnx_model.as_posix()}...')
    if atol is None:
        atol = config.ATOL_FOR_VALIDATION
    if 'diffusers' in str(reference_model.__class__) and (not is_diffusers_available()):
        raise ImportError('The pip package `diffusers` is required to validate stable diffusion ONNX models.')
    framework = 'pt' if is_torch_available() and isinstance(reference_model, nn.Module) else 'tf'
    if input_shapes is None:
        input_shapes = {}
    reference_model_inputs = config.generate_dummy_inputs(framework=framework, **input_shapes)
    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    if device.startswith('cuda'):
        provider = 'CUDAExecutionProvider'
    else:
        provider = 'CPUExecutionProvider'
    session = PickableInferenceSession(onnx_model.as_posix(), sess_options=session_options, providers=[provider])
    all_onnx_outputs = {output.name for output in session.get_outputs()}
    config_outputs = set(config.outputs)
    if all_onnx_outputs != config_outputs:
        if len(all_onnx_outputs) > len(config_outputs):
            diff = all_onnx_outputs - config_outputs
        else:
            diff = config_outputs - all_onnx_outputs
        raise OutputMatchError(f'The exported ONNX model does not have the exact same outputs as what is provided in {config.__class__.__name__}. Difference: {', '.join(diff)}')
    all_config_dynamic_axes_names = set()
    for input_ in config.inputs.values():
        all_config_dynamic_axes_names |= set(input_.values())
    for output in config.outputs.values():
        all_config_dynamic_axes_names |= set(output.values())
    for node in session.get_outputs():
        for idx, axis in enumerate(node.shape):
            if isinstance(axis, str) and axis not in all_config_dynamic_axes_names:
                raise DynamicAxisNameError(f'The axis {idx} of input / output node called {node.name} has an unknown name: {axis}')
    if is_torch_available() and isinstance(reference_model, nn.Module):
        reference_model.to(device)
        for key, value in reference_model_inputs.items():
            reference_model_inputs[key] = recursive_to_device(value=value, device=device)
    copy_reference_model_inputs = copy.deepcopy(reference_model_inputs)
    copy_reference_model_inputs = config.rename_ambiguous_inputs(copy_reference_model_inputs)
    with config.patch_model_for_export(reference_model, model_kwargs=model_kwargs):
        if is_torch_available() and isinstance(reference_model, nn.Module):
            with torch.inference_mode():
                ref_outputs = reference_model(**copy_reference_model_inputs)
        else:
            ref_outputs = reference_model(**copy_reference_model_inputs)
    ref_outputs_dict = {}
    for name, value in ref_outputs.items():
        if name == 'past_key_values':
            name = 'present'
        if isinstance(value, (list, tuple)):
            onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
            value = config.flatten_output_collection_property(onnx_output_name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    reference_ort_inputs = config.generate_dummy_inputs_for_validation(reference_model_inputs, onnx_input_names=onnx_input_names)
    onnx_inputs = {}
    for name, value in reference_ort_inputs.items():
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            onnx_inputs.update({tensor_name: pt_tensor.cpu().numpy() for tensor_name, pt_tensor in value.items()})
        elif isinstance(value, dict):
            onnx_inputs.update({tensor_name: pt_tensor.cpu().numpy() for tensor_name, pt_tensor in value.items()})
        else:
            onnx_inputs[name] = value.cpu().numpy()
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)
    onnx_to_torch = {v: k for k, v in config.torch_to_onnx_output_map.items()}
    onnx_named_outputs = [onnx_to_torch.get(k, k) for k in onnx_named_outputs]
    ref_outputs_set, onnx_outputs_set = (set(ref_outputs_dict.keys()), set(onnx_named_outputs))
    if not onnx_outputs_set.issubset(ref_outputs_set):
        raise OutputMatchError(f'ONNX model output names do not match reference model output names.\nReference model output names: {ref_outputs_set}\nONNX model output names: {onnx_outputs_set}\nDifference: {onnx_outputs_set.difference(ref_outputs_set)}')
    else:
        onnx_output_names = ', '.join(onnx_outputs_set)
        logger.info(f'\t-[✓] ONNX model output names match reference model ({onnx_output_names})')
    if 'diffusers' in str(reference_model.__class__) and (not is_diffusers_available()):
        raise ImportError('The pip package `diffusers` is required to validate stable diffusion ONNX models.')
    shape_failures = []
    value_failures = []
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        if is_torch_available() and isinstance(reference_model, nn.Module):
            ref_value = ref_outputs_dict[name].detach().cpu().numpy()
        else:
            ref_value = ref_outputs_dict[name].cpu().numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')
        if not ort_value.shape == ref_value.shape:
            logger.error(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            shape_failures.append((name, ref_value.shape, ort_value.shape))
        else:
            logger.info(f'\t\t-[✓] {ort_value.shape} matches {ref_value.shape}')
        try:
            if not np.allclose(ref_value, ort_value, atol=atol):
                max_diff = np.amax(np.abs(ref_value - ort_value))
                logger.error(f'\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})')
                value_failures.append((name, max_diff))
            else:
                logger.info(f'\t\t-[✓] all values close (atol: {atol})')
        except Exception:
            pass
    if shape_failures:
        msg = '\n'.join((f'- {t[0]}: got {t[1]} (reference) and {t[2]} (ONNX)' for t in shape_failures))
        raise ShapeError(f'Output shapes do not match between reference model and ONNX exported model:\n{msg}')
    if value_failures:
        msg = '\n'.join((f'- {t[0]}: max diff = {t[1]}' for t in value_failures))
        atol_msg = f'The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance {atol}:\n{msg}'
        if isinstance(config, SpeechT5OnnxConfig):
            atol_msg += '\nIMPORTANT NOTE: SpeechT5 uses a dropout at inference and the output validation of ONNX Runtime inference vs PyTorch is expected to fail. Reference: https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L727'
        raise AtolError(atol_msg)