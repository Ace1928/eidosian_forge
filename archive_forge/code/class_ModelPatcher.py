import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
class ModelPatcher:

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None):
        self._model = model
        patching_specs = config.PATCHING_SPECS
        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)
        self.orig_forward_name = 'forward' if hasattr(self._model, 'forward') else 'call'
        self.orig_forward = getattr(self._model, self.orig_forward_name)
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        if config.__class__.__name__ == 'OnnxConfigWithLoss':
            self.real_config = config._onnx_config
        else:
            self.real_config = config
        allow_past_in_outputs = hasattr(self.real_config, 'use_past') and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)
            outputs = self.orig_forward(*args, **kwargs)
            filterd_outputs = {}
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                    if onnx_output_name in config.outputs or (allow_past_in_outputs and name.startswith('past_key_values')) or any((key.startswith(onnx_output_name) for key in config.outputs.keys())):
                        filterd_outputs[name] = value
            elif isinstance(outputs, (list, tuple)):
                outputs_list = list(config.outputs.keys())
                dict(zip(outputs_list, outputs))
            else:
                if len(config.outputs) > 1:
                    num_outputs = len(config.outputs)
                    outputs_str = ', '.join(config.outputs.keys())
                    raise ValueError(f'config.outputs should have only one outputs, but it has {num_outputs} keys: {outputs_str}')
                else:
                    name = list(config.outputs.keys())[0]
                    filterd_outputs[name] = outputs
                name = list(config.outputs.keys())[0]
                filterd_outputs[name] = outputs
            return filterd_outputs
        self.patched_forward = patched_forward

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    def __enter__(self):
        self.patch_ops()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)

    def __call__(self, *args, **kwargs):
        if getattr(self._model, self.orig_forward_name) is self.orig_forward:
            logger.warning('Running the non-patched model')
        return self._model(*args, **kwargs)