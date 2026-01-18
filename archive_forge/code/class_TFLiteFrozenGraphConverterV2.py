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
class TFLiteFrozenGraphConverterV2(TFLiteConverterBaseV2):
    """Converts the given frozen graph into TensorFlow Lite model."""

    def __init__(self, funcs, trackable_obj=None):
        """Constructor for TFLiteConverter.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
        super(TFLiteFrozenGraphConverterV2, self).__init__()
        self._funcs = funcs
        self._trackable_obj = trackable_obj
        self.experimental_lower_to_saved_model = True

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_CONCRETE_FUNCTION)
    def _freeze_concrete_function(self):
        """Convert the given ConcreteFunction to frozen graph.

    Returns:
      graph_def: The frozen GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      frozen_func: The frozen ConcreteFunction.

    Raises:
      ValueError: none or multiple ConcreteFunctions provided.
    """
        if len(self._funcs) == 0:
            raise ValueError('No ConcreteFunction is specified.')
        if len(self._funcs) > 1:
            raise ValueError('This converter can only convert a single ConcreteFunction. Converting multiple functions is under development.')
        frozen_func, graph_def = _convert_to_constants.convert_variables_to_constants_v2_as_graph(self._funcs[0], lower_control_flow=False)
        input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != _dtypes.resource]
        output_tensors = frozen_func.outputs
        return (graph_def, input_tensors, output_tensors, frozen_func)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL)
    def _convert_concrete_functions_to_saved_model(self, output_dir):
        """Save concrete functions to the SavedModel format.

    Args:
      output_dir: The output directory to save the SavedModel.

    Returns:
      graph_def: The frozen GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
    """
        if len(self._funcs) == 0:
            raise ValueError('No ConcreteFunction is specified.')
        if not self.experimental_lower_to_saved_model:
            return (None, None, None)
        if not self._trackable_obj or isinstance(self._trackable_obj, (_function.ConcreteFunction, _def_function.Function)):
            return (None, None, None)
        signatures = {}
        signature_keys = []
        try:
            if len(self._funcs) == 1:
                signatures[_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = self._funcs[0]
                signature_keys = [_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            else:
                for func in self._funcs:
                    signatures[func.graph.name] = func
                    signature_keys.append(func.graph.name)
            _saved_model.save(self._trackable_obj, output_dir, signatures=signatures, options=_save_options.SaveOptions(save_debug_info=True))
        except Exception:
            return (None, None, None)
        self.saved_model_dir = output_dir
        self._saved_model_tags = set([_tag_constants.SERVING])
        self._saved_model_exported_names = signature_keys
        self._parse_saved_model_args(always_enable_saved_model_import=True)
        if self.saved_model_dir:
            graph_def, input_tensors, output_tensors = self._load_saved_model(self.saved_model_dir, self._saved_model_tags)
            self._trackable_obj = _load(self.saved_model_dir, self._saved_model_tags)
            return (graph_def, input_tensors, output_tensors)
        return (None, None, None)

    def _convert_as_saved_model(self):
        """Converts the given concrete functions as a saved model format.

    Returns:
      The converted data in serialized format.
    """
        temp_dir = tempfile.mkdtemp()
        try:
            graph_def, input_tensors, _ = self._convert_concrete_functions_to_saved_model(temp_dir)
            if self.saved_model_dir:
                self._validate_inputs(graph_def, input_tensors)
                return self._convert_from_saved_model(graph_def)
        finally:
            shutil.rmtree(temp_dir, True)
        return None

    @_export_metrics
    def convert(self):
        """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
        if self.experimental_lower_to_saved_model:
            saved_model_convert_result = self._convert_as_saved_model()
            if saved_model_convert_result:
                return saved_model_convert_result
        graph_def, input_tensors, output_tensors, frozen_func = self._freeze_concrete_function()
        graph_def = self._optimize_tf_model(graph_def, input_tensors, output_tensors, frozen_func)
        return super(TFLiteFrozenGraphConverterV2, self).convert(graph_def, input_tensors, output_tensors)