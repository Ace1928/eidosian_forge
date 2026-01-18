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
def _get_base_converter_args(self):
    """Returns the base converter args.

    Returns:
      {key str: val}
    """
    args = {'input_format': constants.TENSORFLOW_GRAPHDEF, 'allow_custom_ops': self.allow_custom_ops, 'debug_info': self._debug_info, 'target_ops': self.target_spec.supported_ops, 'enable_mlir_converter': self.experimental_new_converter, 'select_user_tf_ops': self.target_spec.experimental_select_user_tf_ops, 'supported_backends': self.target_spec.experimental_supported_backends, 'unfold_batchmatmul': self.unfold_batchmatmul, 'legalize_custom_tensor_list_ops': self.legalize_custom_tensor_list_ops, 'lower_tensor_list_ops': self._experimental_lower_tensor_list_ops, 'unfold_large_splat_constant': self._experimental_unfold_large_splat_constant, 'default_to_single_batch_in_tensor_list_ops': self._experimental_default_to_single_batch_in_tensor_list_ops, 'tf_quantization_mode': self._experimental_tf_quantization_mode, 'experimental_enable_resource_variables': self.experimental_enable_resource_variables, 'enable_dynamic_update_slice': self._experimental_enable_dynamic_update_slice, 'preserve_assert_op': self._experimental_preserve_assert_op, 'guarantee_all_funcs_one_use': self._experimental_guarantee_all_funcs_one_use, 'allow_all_select_tf_ops': self._experimental_allow_all_select_tf_ops, 'disable_fuse_mul_and_fc': self._experimental_disable_fuse_mul_and_fc, 'quantization_options': self._experimental_quantization_options, 'mlir_dump_dir': self.mlir_dump_dir, 'mlir_dump_pass_regex': self.mlir_dump_pass_regex, 'mlir_dump_func_regex': self.mlir_dump_func_regex, 'mlir_enable_timing': self.mlir_enable_timing, 'mlir_print_ir_before': self.mlir_print_ir_before, 'mlir_print_ir_after': self.mlir_print_ir_after, 'mlir_print_ir_module_scope': self.mlir_print_ir_module_scope, 'mlir_elide_elementsattrs_if_larger': self.mlir_elide_elementsattrs_if_larger, 'use_buffer_offset': self._experimental_use_buffer_offset}
    if self.saved_model_dir:
        args.update({'saved_model_dir': self.saved_model_dir, 'saved_model_version': self._saved_model_version, 'saved_model_tags': self._saved_model_tags, 'saved_model_exported_names': self._saved_model_exported_names})
    if self._experimental_quantization_options:
        logging.warning('Configs from custom methods in experimental_quantization_options may not produce a valid tflite model. Note that currently this option only supports StableHLO path. Setting this option in TFLite path will be a no-op.')
    return args