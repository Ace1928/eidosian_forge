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
class TFLiteConverterBaseV1(TFLiteConverterBase):
    """Converter subclass to share functionality between V1 converters."""

    def __init__(self, experimental_debug_info_func):
        """Constructor for TFLiteConverter.

    Args:
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.
    """
        super(TFLiteConverterBaseV1, self).__init__()
        self.inference_type = _dtypes.float32
        self.inference_input_type = None
        self.inference_output_type = None
        self.output_format = constants.TFLITE
        self.quantized_input_stats = {}
        self.default_ranges_stats = None
        self.drop_control_dependency = True
        self.reorder_across_fake_quant = False
        self.change_concat_input_ranges = False
        self.dump_graphviz_dir = None
        self.dump_graphviz_video = False
        self.conversion_summary_dir = None
        self._debug_info_func = experimental_debug_info_func
        self._metadata.environment.apiVersion = 1

    def __setattr__(self, name, value):
        if name == 'post_training_quantize':
            warnings.warn('Property %s is deprecated, please use optimizations=[Optimize.DEFAULT] instead.' % name)
            if value:
                self.optimizations = [Optimize.DEFAULT]
            else:
                self.optimizations = []
            return
        if name == 'target_ops':
            warnings.warn('Property %s is deprecated, please use target_spec.supported_ops instead.' % name)
            self.target_spec.supported_ops = value
            return
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        if name == 'post_training_quantize':
            warnings.warn('Property %s is deprecated, please use optimizations=[Optimize.DEFAULT] instead.' % name)
            return Optimize.DEFAULT in set(self.optimizations)
        if name == 'target_ops':
            warnings.warn('Property %s is deprecated, please use target_spec.supported_ops instead.' % name)
            return self.target_spec.supported_ops
        return object.__getattribute__(self, name)

    def _validate_quantized_input_stats(self, converter_kwargs, quant_mode):
        """Ensure the `quantized_input_stats` flag is provided if required."""
        quantized_types = frozenset({_dtypes.int8, _dtypes.uint8})
        requires_quantized_input_stats = (converter_kwargs['inference_type'] in quantized_types or converter_kwargs['inference_input_type'] in quantized_types) and (not quant_mode.is_post_training_integer_quantization())
        if requires_quantized_input_stats and (not converter_kwargs['quantized_input_stats']):
            raise ValueError('The `quantized_input_stats` flag must be defined when either `inference_type` flag or `inference_input_type` flag is set to tf.int8 or tf.uint8. Currently, `inference_type={}` and `inference_input_type={}`.'.format(_get_tf_type_name(converter_kwargs['inference_type']), _get_tf_type_name(converter_kwargs['inference_input_type'])))

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.VALIDATE_INPUTS)
    def _validate_inputs(self, input_tensors, quantized_input_stats):
        """Validate input parameters.

    Args:
      input_tensors: List of input tensors.
      quantized_input_stats: Map of input tensor names to a tuple of floats
        representing the mean and standard deviation of the training data.

    Raises:
      ValueError:
        Input shape is not specified.
        Quantization input stats is required but not provided.
    """
        if not self._is_unknown_shapes_allowed() and self._has_valid_tensors():
            for tensor in input_tensors:
                shape = tensor.shape
                if not shape:
                    raise ValueError("Provide an input shape for input array '{0}'.".format(_get_tensor_name(tensor)))
                shape_list = shape.as_list()
                if None in shape_list[1:]:
                    raise ValueError("None is only supported in the 1st dimension. Tensor '{0}' has invalid shape '{1}'.".format(_get_tensor_name(tensor), shape_list))
                elif shape_list and shape_list[0] is None:
                    self._set_batch_size(batch_size=1)
        if quantized_input_stats:
            self._quantized_stats = []
            invalid_stats = []
            for name in self.get_input_arrays():
                if name in quantized_input_stats:
                    self._quantized_stats.append(quantized_input_stats[name])
                else:
                    invalid_stats.append(name)
            if invalid_stats:
                raise ValueError("Quantization input stats are not available for input tensors '{0}'.".format(','.join(invalid_stats)))
        else:
            self._quantized_stats = None

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.OPTIMIZE_TF_MODEL)
    def _optimize_tf_model(self, graph_def, input_tensors, output_tensors, quant_mode):
        """Run a Grappler pass to optimize the TensorFlow graph.

    Args:
      graph_def: Frozen GraphDef to be optimized.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      quant_mode: the quantization mode.

    Returns:
      The optimized TensorFlow graph.
    """
        if self.saved_model_dir or quant_mode.is_quantization_aware_trained_model():
            return graph_def
        try:
            graph = _convert_to_constants.disable_lower_using_switch_merge(graph_def)
            optimized_graph = _run_graph_optimizations(graph, input_tensors, output_tensors, config=self._grappler_config(['function']))
            return optimized_graph
        except Exception:
            return graph_def

    def convert(self):
        """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
        self._validate_inputs(self._input_tensors, self.quantized_input_stats)
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, self._graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        optimized_graph = self._optimize_tf_model(self._graph_def, self._input_tensors, self._output_tensors, quant_mode)
        self._debug_info = _get_debug_info(self._debug_info_func, optimized_graph)
        converter_kwargs = self._get_base_converter_args()
        converter_kwargs.update(quant_mode.converter_flags(self.inference_type, self.inference_input_type))
        converter_kwargs.update({'output_format': self.output_format, 'quantized_input_stats': self._quantized_stats, 'default_ranges_stats': self.default_ranges_stats, 'drop_control_dependency': self.drop_control_dependency, 'reorder_across_fake_quant': self.reorder_across_fake_quant, 'change_concat_input_ranges': self.change_concat_input_ranges, 'dump_graphviz_dir': self.dump_graphviz_dir, 'dump_graphviz_video': self.dump_graphviz_video, 'conversion_summary_dir': self.conversion_summary_dir})
        self._validate_quantized_input_stats(converter_kwargs, quant_mode)
        if not self.experimental_new_converter:
            logging.warning('Please consider switching to the new converter by setting experimental_new_converter=True. The old converter is deprecated.')
        else:
            logging.info('Using experimental converter: If you encountered a problem please file a bug. You can opt-out by setting experimental_new_converter=False')
        if self._has_valid_tensors():
            result = _convert_graphdef(input_data=optimized_graph, input_tensors=self._input_tensors, output_tensors=self._output_tensors, **converter_kwargs)
        else:
            result = _convert_graphdef_with_arrays(input_data=optimized_graph, input_arrays_with_shape=self._input_arrays_with_shape, output_arrays=self._output_arrays, control_output_arrays=self._control_output_arrays, **converter_kwargs)
        return self._optimize_tflite_model(result, quant_mode, quant_io=self.experimental_new_quantizer)

    def get_input_arrays(self):
        """Returns a list of the names of the input tensors.

    Returns:
      List of strings.
    """
        if self._has_valid_tensors():
            return [_get_tensor_name(tensor) for tensor in self._input_tensors]
        else:
            return [name for name, _ in self._input_arrays_with_shape]

    def _has_valid_tensors(self):
        """Checks if the input and output tensors have been initialized.

    Returns:
      Bool.
    """
        return self._input_tensors is not None and self._output_tensors

    def _set_batch_size(self, batch_size):
        """Sets the first dimension of the input tensor to `batch_size`.

    Args:
      batch_size: Batch size for the model. Replaces the first dimension of an
        input size array if undefined. (default 1)

    Raises:
      ValueError: input_tensor is not defined.
    """
        if not self._has_valid_tensors():
            raise ValueError('The batch size cannot be set for this model. Please use input_shapes parameter.')
        for tensor in self._input_tensors:
            shape = tensor.shape.as_list()
            if shape[0] is None:
                shape[0] = batch_size
                tensor.set_shape(shape)

    def _is_unknown_shapes_allowed(self):
        if _is_ophint_converted(self._graph_def):
            return False
        if not super(TFLiteConverterBaseV1, self)._is_unknown_shapes_allowed():
            return False
        if self.conversion_summary_dir:
            logging.warning('`conversion_summary_dir` does not work with unknown shapes. Graphs with unknown shapes might be different than when this flag is disabled.')
            return False
        return True

    def _save_conversion_params_metric(self):
        self._collected_converter_params.update({'output_format': self.output_format, 'default_ranges_stats': self.default_ranges_stats, 'drop_control_dependency': self.drop_control_dependency, 'reorder_across_fake_quant': self.reorder_across_fake_quant, 'change_concat_input_ranges': self.change_concat_input_ranges, 'dump_graphviz_dir': self.dump_graphviz_dir, 'dump_graphviz_video': self.dump_graphviz_video, 'conversion_summary_dir': self.conversion_summary_dir})
        super(TFLiteConverterBaseV1, self)._save_conversion_params_metric(self._graph_def, self.inference_type, self.inference_input_type)