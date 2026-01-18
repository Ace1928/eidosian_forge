import enum
import functools
import pprint
import shutil
import sys
import tempfile
import time
import warnings
from absl import logging
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op  # pylint: disable=unused-import
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metdata_fb
from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.convert import convert_graphdef as _convert_graphdef
from tensorflow.lite.python.convert import convert_graphdef_with_arrays as _convert_graphdef_with_arrays
from tensorflow.lite.python.convert import convert_jax_hlo as _convert_jax_hlo
from tensorflow.lite.python.convert import convert_saved_model as _convert_saved_model
from tensorflow.lite.python.convert import ConverterError  # pylint: disable=unused-import
from tensorflow.lite.python.convert import deduplicate_readonly_buffers as _deduplicate_readonly_buffers
from tensorflow.lite.python.convert import mlir_quantize as _mlir_quantize
from tensorflow.lite.python.convert import mlir_sparsify as _mlir_sparsify
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.convert import toco_convert  # pylint: disable=unused-import
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import load_delegate  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import OpResolverType  # pylint: disable=unused-import
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.lite.python.op_hint import is_ophint_converted as _is_ophint_converted
from tensorflow.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.python.util import _xla_computation
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func
from tensorflow.lite.python.util import convert_debug_info_func as _convert_debug_info_func
from tensorflow.lite.python.util import freeze_graph as _freeze_graph
from tensorflow.lite.python.util import get_debug_info as _get_debug_info
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_sparsity_modes as _get_sparsity_modes
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import get_tf_type_name as _get_tf_type_name
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import model_input_signature as _model_input_signature
from tensorflow.lite.python.util import modify_model_io_type as _modify_model_io_type
from tensorflow.lite.python.util import populate_conversion_metadata as _populate_conversion_metadata
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.lite.python.util import set_tensor_shapes as _set_tensor_shapes
from tensorflow.lite.python.util import trace_model_call as _trace_model_call
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugger  # pylint: disable=unused-import
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugOptions  # pylint: disable=unused-import
from tensorflow.python import saved_model as _saved_model
from tensorflow.python.client import session as _session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader_impl as _loader_impl
from tensorflow.python.saved_model import save_options as _save_options
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants
from tensorflow.python.saved_model.load import load as _load
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info as _parse_saved_model_with_debug_info
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util import keras_deps
from tensorflow.python.util.tf_export import tf_export as _tf_export
class TFLiteJaxConverterV2(TFLiteConverterBaseV2):
    """Converts the given jax model into TensorFlow Lite model."""

    def __init__(self, serving_funcs, inputs):
        """Constructor for TFLiteConverter.

    Args:
      serving_funcs: A list functions of the serving func of the jax module, the
        model params should already be inlined. (e.g., `serving_func =
        functools.partial(model, params=params)`)
      inputs: Array of input tensor placeholders tuple,s like `jnp.zeros`. For
        example, wrapped in an array like "[('input1', input1), ('input2',
        input2)]]".

    Jax functions are polymorphic, for example:

    ```python
    def add(a, b):
      return a + b
    ```

    Will yield different computations if different input signatures are passed
    in: Pass `add(10.0, 20.0)` will yield a scalar `add` while pass
    `add(np.random((100, 1)), np.random(100, 100))` will yield a broadcasting
    add.  We will need the input information to do tracing for the converter
    to properly convert the model. So it's important to pass in the desired
    `input placeholders` with the correct input shape/type.

    In the converted tflite model, the function name will be default to "main",
    the output names will be the traced outputs. The output ordering shall
    match the serving function.
    """
        super(TFLiteJaxConverterV2, self).__init__()
        self._serving_funcs = serving_funcs
        self._inputs = inputs

    @_export_metrics
    def convert(self):
        """Converts a Jax serving func based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ImportError:
        If cannot import the xla_computation from jax.
      ValueError:
        No serving function is specified.
        Input tensors are not specified.
        The truth value of an array with more than one element is ambiguous.
        Failed to convert the given Jax function to hlo.
    """
        if not _xla_computation:
            raise ImportError('Cannot import xla_computation from jax.')
        if not self._serving_funcs:
            raise ValueError('No serving func is specified.')
        if not self._inputs:
            raise ValueError('Input tensors are not specified.')
        if len(self._inputs) != len(self._serving_funcs):
            msg = 'Input tensor mapping len {} does not match serving func len {}.'.format(len(self._inputs), len(self._serving_funcs))
            raise ValueError(msg)
        if not isinstance(self._inputs, (tuple, list)):
            raise ValueError('Input tensors should be pass in a tuple list wrapped in an array.')
        if len(self._serving_funcs) > 1:
            raise ValueError('Currently only support single serving function.')
        if not isinstance(self._inputs[0], (tuple, list)):
            raise ValueError('The input placeholders are not a dictionary.')
        input_names = []
        ordered_inputs = []
        for input_name, tensor in self._inputs[0]:
            input_names.append(input_name)
            ordered_inputs.append(tensor)
        try:
            xla_compuation = _xla_computation(self._serving_funcs[0], backend='cpu')
            hlo_proto = xla_compuation(*ordered_inputs).as_serialized_hlo_module_proto()
        except Exception:
            raise ValueError('Failed to convert the given Jax function to hlo.')
        converter_kwargs = {'input_content': hlo_proto, 'input_names': input_names, 'is_proto_format': True}
        converter_kwargs.update(self._get_base_converter_args())
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, None)
        self._validate_inference_input_output_types(quant_mode)
        converter_kwargs.update(quant_mode.converter_flags())
        result = _convert_jax_hlo(**converter_kwargs)
        return self._optimize_tflite_model(result, quant_mode, quant_io=self.experimental_new_quantizer)